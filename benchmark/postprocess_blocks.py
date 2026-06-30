import json
import os
from pathlib import Path

from openai import OpenAI
from google import genai
from anthropic import Anthropic
from pydantic import ValidationError

from rich.console import Console
console = Console()

from utils import (
    parse_json,
    remove_wrapper,
    get_versioned_path,
    LONG_FORM_LABEL_CATEGORIES,
    SHORT_FORM_LABEL_CATEGORIES,
)

from models import LayoutElement
from release_loader import load_forms


class BatchPostprocessor:
    """Postprocessing pipeline for batch API results (for DLA)"""

    def __init__(
        self, model_family, task_type, label_set_variant,
        model_name, release_root, dataset, prompt_variant, thinking_mode,
        lang, batch_size, batch_num,
    ):
        self.model_family = model_family
        self.task_type = task_type
        self.label_set_variant = label_set_variant
        if model_family == "openai":
            self.client = OpenAI()
        elif model_family == "gemini":
            self.client = genai.Client()
        elif model_family == "claude":
            self.client = Anthropic()
        else:
            self.client = None

        all_form_annotations = load_forms(release_root, task_type)

        start_idx = (batch_num - 1) * batch_size
        end_idx = start_idx + min(batch_size, len(all_form_annotations[start_idx:]))
        reasoning_state = "nonreasoning" if thinking_mode == "off" else "reasoning"

        batch_stem = f"{task_type}_{dataset}_{start_idx+1}_{end_idx}_{prompt_variant}_{label_set_variant}_labelset_{reasoning_state}"
        batch_dir = f"batchDetails/{task_type}/{lang}/{model_name}"
        self.metadata_file_path = f"{batch_dir}/metadata_{batch_stem}.json"
        self.prediction_file_path = Path(f"{batch_dir}/predictions_{batch_stem}.jsonl")
        self.error_file_path = Path(f"{batch_dir}/errors_{batch_stem}.jsonl")

        output_dir = f"predictions/{dataset}/dla/{lang}/{model_name}/{prompt_variant}"
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = get_versioned_path(
            f"{output_dir}/{task_type}_{start_idx+1}_{end_idx}_{label_set_variant}_labelset_{reasoning_state}_predictions.json"
        )

    @staticmethod
    def parse_inference_id(inference_id, task_type):
        if task_type == "dla":
            form_idx, _, page_idx = inference_id.partition("_page_")
            return int(form_idx), int(page_idx)
        elif task_type == "kie":
            form_idx, _, residue = inference_id.partition("_page_")
            page_idx, _, pair_idx = residue.partition("_pair_")
            return int(form_idx), int(page_idx), int(pair_idx)
        return

    def fetch_batch_status(self, batch_id):
        if self.model_family == "openai":
            return self.client.batches.retrieve(batch_id)
        elif self.model_family == "gemini":
            return self.client.batches.get(name=batch_id)
        elif self.model_family == "claude":
            return self.client.messages.batches.retrieve(batch_id)

    def retrieve_output(self, batch_details, prediction_file_path, error_file_path):
        if self.model_family == "claude":
            status = batch_details.processing_status
            if status not in {"ended"}:
                print(f"The inference process is not completed yet. CURRENT STATUS: {status}")
            else:
                if batch_details.results_url:
                    with open(prediction_file_path, "w", encoding="utf-8") as f:
                        results = self.client.messages.batches.results(batch_details.id)
                        for item in results:
                            obj = item.model_dump() if hasattr(item, "model_dump") else item
                            obj = json.dumps(obj, ensure_ascii=False)
                            f.write(obj + "\n")
        elif self.model_family == "openai":
            status = batch_details.status
            # possible openai batch status: validating, in_progress, finalizing, completed, failed
            if status not in {"completed", "failed"}:
                print(f"The inference process is not completed yet. CURRENT STATUS: {status}")
            else:
                if batch_details.error_file_id:
                    file_content = self.client.files.content(batch_details.error_file_id)
                    Path(error_file_path).write_bytes(file_content.read())
                if batch_details.output_file_id:
                    file_content = self.client.files.content(batch_details.output_file_id)
                    Path(prediction_file_path).write_bytes(file_content.read())
        elif self.model_family == "gemini":
            status = batch_details.state.name
            # possible gemini batch status: JOB_STATE_PENDING, JOB_STATE_RUNNING, JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELLED, JOB_STATE_EXPIRED
            if status not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                print(f"The inference process is not completed yet. CURRENT STATUS: {status}")
            else:
                if batch_details.dest and batch_details.dest.file_name:
                    result_file_name = batch_details.dest.file_name
                    Path(prediction_file_path).write_bytes(self.client.files.download(file=result_file_name))
        elif self.model_family == "qwen":
            pass

    def _id_key(self):
        if self.model_family in {"openai", "claude"}:
            return "custom_id"
        elif self.model_family == "gemini":
            return "key"

    def sort_predictions(self, raw_preds):
        sorting_key_str = self._id_key()
        sorted_preds = sorted(raw_preds, key=lambda raw_pred: self.parse_inference_id(raw_pred[sorting_key_str], self.task_type))
        return sorted_preds

    def standardize_predictions(self, preds):
        id_key = self._id_key()
        standardized_preds = [{**pred, "inference_id": pred[id_key]} for pred in preds]
        if self.model_family == "openai":
            standardized_preds = [
                {
                    **sp,
                    "preds": next(
                        (
                            c.get("text", "")
                            for o in sp.get("response", {}).get("body", {}).get("output", [])
                            if isinstance(o, dict) and o.get("type") == "message"
                            for c in o.get("content", [])
                            if isinstance(c, dict) and "text" in c
                        ),
                        ""
                    )
                }
                for sp in standardized_preds
            ]
        elif self.model_family == "gemini":
            standardized_preds = [
                {
                    **sp,
                    "preds": next(
                        (
                            p["text"]
                            for p in (
                                (sp.get("response", {}).get("candidates") or [{}])[0]
                                .get("content", {})
                                .get("parts", [])
                            )
                            if isinstance(p, dict) and "text" in p
                        ),
                        ""
                    ),
                }
                for sp in standardized_preds
            ]
        elif self.model_family == "claude":
            standardized_preds = [
                {
                    **sp,
                    "preds": "\n".join(
                        t for t in [
                            c.get("text", "")
                            for c in sp.get("result", {}).get("message", {}).get("content", [])
                            if isinstance(c, dict) and c.get("type") == "text" and isinstance(c.get("text", ""), str)
                        ]
                        if t.strip()
                    ),
                }
                for sp in standardized_preds
            ]
        return standardized_preds

    def parse_predictions(self, all_preds):
        parsed_cnt = 0
        parsed_preds = []
        for idx, preds in enumerate(all_preds):
            try:
                json_preds = parse_json(preds["preds"])
                json_preds = remove_wrapper(json_preds)
                if isinstance(json_preds, dict):
                    json_preds = [json_preds]
                parsed_cnt += len(json_preds)
                all_preds[idx]["json_preds"] = json_preds
                parsed_preds.append(all_preds[idx])
            except Exception as e:
                console.print(f"Parsing Error: {str(e)}", style="bold red")
                continue
        return parsed_preds, parsed_cnt

    def validate_predictions(self, all_preds):
        valid_cnt, invalid_cnt = 0, 0
        if self.label_set_variant == "full":
            categories = {category.strip().lower() for category in LONG_FORM_LABEL_CATEGORIES}
        elif self.label_set_variant == "reduced":
            categories = {category.strip().lower() for category in SHORT_FORM_LABEL_CATEGORIES}
        else:
            raise ValueError(f"Unknown label_set_variant: {self.label_set_variant}")

        for idx, preds in enumerate(all_preds):
            validated_preds = []
            invalidated_preds = []
            json_preds = preds.get("json_preds", [])
            if json_preds:
                for pred in json_preds:
                    try:
                        val = LayoutElement.model_validate(pred, context={"categories": categories})
                        validated_preds.append(val.model_dump(by_alias=True))
                    except ValidationError as e:
                        invalidated_preds.append({
                            "id": pred.get("id"),
                            "errors": e.errors(),
                        })
                all_preds[idx]["validated_preds"] = validated_preds
                all_preds[idx]["invalidated_preds"] = invalidated_preds
                all_preds[idx]["consistent_struct_count"] = len(validated_preds)
                all_preds[idx]["inconsistent_struct_count"] = len(invalidated_preds)
                valid_cnt += all_preds[idx]["consistent_struct_count"]
                invalid_cnt += all_preds[idx]["inconsistent_struct_count"]
        return all_preds, valid_cnt, invalid_cnt

    def group_by_form(self, finalized_predictions):
        formatted_predictions = []
        prev_form_idx = -1
        form_details = dict()
        for preds in finalized_predictions:
            curr_form_idx, _ = self.parse_inference_id(preds["inference_id"], self.task_type)
            if curr_form_idx != prev_form_idx:
                if prev_form_idx != -1:
                    formatted_predictions.append(form_details)
                form_details = {"form_id": int(curr_form_idx)}
                form_details["preds"] = [preds.get("validated_preds", [])]
            else:
                form_details["preds"].append(preds.get("validated_preds", []))
            prev_form_idx = curr_form_idx

        if form_details:
            formatted_predictions.append(form_details)
        return formatted_predictions
