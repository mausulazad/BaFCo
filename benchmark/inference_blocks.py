from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

from rich.console import Console
console = Console()

from utils import (
    load_form_pages,
    encode_image,
    parse_json,
    remove_wrapper,
    OPENROUTER_MODEL_PATH_MAP,
    NATIVE_MODEL_PATH_MAP,
)
from release_loader import load_forms, subset_dir, resolve_release_image

from prompts import build_prompt

from models import LayoutElement, LayoutAnalysisOutput

@dataclass
class ModelConfig:
    tuning_free: bool
    model_family: str

@dataclass
class InferenceConfig:
    use_api: bool
    temperature: float
    top_p: float
    thinking_mode: str
    use_openrouter: bool = True

class FileProcessor:
    def __init__(self, task_type, model_name, release_root, batch_size, batch_num, prompt_variant, label_set_variant, thinking_mode, lang="all", dataset_name="bafco"):
        self.task_type = task_type
        self.model_name = model_name
        self.prompt_variant = prompt_variant
        self.label_set_variant = label_set_variant
        self.dataset_name = dataset_name
        self.release_root = release_root
        self.subset_dir = subset_dir(release_root, task_type)
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.lang = lang
        self.reasoning_state = "nonreasoning" if thinking_mode == "off" else "reasoning"
        self._build_file_paths()

    def _build_file_paths(self):
        output_dir = f"batchDetails/{self.task_type}/{self.lang}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)

        dataset = load_forms(self.release_root, self.task_type)

        start_idx = (self.batch_num - 1) * self.batch_size
        end_idx = start_idx + min(self.batch_size, len(dataset[start_idx:]))
        self.dataset = dataset[start_idx : end_idx]
        self.start_idx = start_idx + 1
        self.end_idx = end_idx

        slug = f"{self.task_type}_{self.dataset_name}_{start_idx+1}_{end_idx}_{self.prompt_variant}_{self.label_set_variant}_labelset_{self.reasoning_state}"
        self.batch_input_file_path = os.path.join(output_dir, f"batch_input_{slug}.jsonl")
        self.metadata_file_path = os.path.join(output_dir, f"metadata_{slug}.json")

    def load_annotations(self) -> list:
        return self.dataset

    def get_image(self, form: dict, page_idx: int) -> bytes:
        """Base64 image for one page, resolved from the form's ``local_images``. None if missing."""
        if not 0 <= page_idx < len(form.get("local_images", [])):
            return None
        path = resolve_release_image(self.subset_dir, form, page_idx)
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return None
        return encode_image(path)

    def get_prediction_path(self) -> str:
        if self.task_type == "kie":
            path = f"predictions/{self.dataset_name}/kie/all/{self.model_name}_{self.prompt_variant}_{self.reasoning_state}_{self.start_idx}_{self.end_idx}.jsonl"
        elif self.task_type == "dla":
            path = f"predictions/{self.dataset_name}/dla/{self.lang}/{self.model_name}/{self.prompt_variant}/{self.task_type}_{self.start_idx}_{self.end_idx}_{self.label_set_variant}_labelset_{self.reasoning_state}_predictions.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


class Inferencer(ABC):
    """Abstract base — TaskRunner only calls .run(), never cares about batch vs sync."""

    def __init__(self, file_processor: FileProcessor, model_config: ModelConfig, inference_config: InferenceConfig):
        self.file_processor = file_processor
        self.model_family = model_config.model_family
        self.temperature = inference_config.temperature
        self.top_p = inference_config.top_p
        self.thinking_mode = inference_config.thinking_mode

    @abstractmethod
    def run(self) -> List[dict]:
        """Execute inference and return predictions."""
        ...


class SyncInferencer(Inferencer):
    """Sequential per-item inference: covers KIE and DLA non-batch (OpenRouter / direct API)."""

    def __init__(self, file_processor: FileProcessor, model_config: ModelConfig, inference_config: InferenceConfig):
        super().__init__(file_processor, model_config, inference_config)
        self.use_openrouter = inference_config.use_openrouter
        path_map = OPENROUTER_MODEL_PATH_MAP if self.use_openrouter else NATIVE_MODEL_PATH_MAP
        self.model_path = path_map[model_config.model_family][file_processor.model_name]
        # max_retries=0 so the SDK's internal retries don't compound outer 3-attempt loop.
        self.or_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            timeout=180.0,
            max_retries=0,
        )
        self.gemini_client = genai.Client()
        self.claude_client = Anthropic()
        self.openai_client = OpenAI()

    def _call_with_timeout(self, fn, *args, timeout: float = 600.0, **kwargs):
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            fut = executor.submit(fn, *args, **kwargs)
            return fut.result(timeout=timeout)
        finally:
            executor.shutdown(wait=False)

    def call(self, prompt: str, image_bytes, schema: dict | None = None) -> str:
        if self.use_openrouter:
            kwargs = dict(
                model=self.model_path,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"}}
                        ]
                    }
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            if self.thinking_mode == "on":
                kwargs["max_tokens"] = 64000
                kwargs["extra_body"] = {"reasoning": {"effort": "high"}}
            elif self.model_path.startswith("google/gemini-3"):
                kwargs["extra_body"] = {"reasoning": {"effort": "low"}}
            else:
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            response = self._call_with_timeout(
                self.or_client.chat.completions.create, timeout=600.0, **kwargs
            )
            return response.choices[0].message.content

        else:
            if self.model_family == "openai":
                reasoning_effort = "high" if self.thinking_mode == "on" else "low"
                response = self.openai_client.responses.create(
                    model=self.model_path,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_bytes}"},
                            ],
                        }
                    ],
                    temperature=self.temperature,
                    reasoning={"effort": reasoning_effort},
                )
                return response.output_text

            elif self.model_family == "claude":
                effort = "high" if self.thinking_mode == "on" else "low"
                response = self.claude_client.messages.create(
                    model=self.model_path,
                    max_tokens=2000,
                    thinking={"type": "adaptive"},
                    output_config={"effort": effort},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_bytes}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                )
                return "\n".join(block.text for block in response.content if block.type == "text")

            elif self.model_family == "gemini":
                img_bytes = base64.b64decode(image_bytes)
                config_kwargs = {"response_mime_type": "application/json"}
                if schema is not None:
                    config_kwargs["response_json_schema"] = schema
                if self.model_path.startswith("gemini-3"):
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")
                response = self.gemini_client.models.generate_content(
                    model=self.model_path,
                    contents=[
                        types.Content(parts=[
                            types.Part(text=prompt),
                            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes)),
                        ])
                    ],
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                return response.text

    def run(self) -> List[dict]:
        if self.file_processor.task_type == "dla":
            return self._run_dla()
        return self._run_kie()

    def _run_dla(self) -> List[dict]:
        fp = self.file_processor
        annotations = fp.load_annotations()
        prediction_path = fp.get_prediction_path()
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

        formatted_predictions: List[dict] = []
        done_ids: set = set()
        if os.path.exists(prediction_path):
            try:
                with open(prediction_path, "r", encoding="utf-8") as f:
                    formatted_predictions = json.load(f)
                formatted_predictions = [
                    p for p in formatted_predictions
                    if any(page for page in p.get("preds", []))
                ]
                done_ids = {p["form_id"] for p in formatted_predictions if "form_id" in p}
                print(f"Resuming — {len(done_ids)} form(s) already done in {prediction_path}, skipping.")
            except Exception as e:
                print(f"Could not parse existing predictions ({type(e).__name__}: {e}); starting fresh.")
                formatted_predictions = []
                done_ids = set()

        total_form_cnt = len(annotations)

        for idx, form_annotations in enumerate(tqdm(annotations, desc="DLA sequential inference"), start=1):
            form_id = form_annotations["id"]
            if form_id in done_ids:
                continue

            form_page_images = load_form_pages(fp.subset_dir, form_annotations)
            form_preds = {"form_id": form_id, "preds": []}
            t_form_start = time.perf_counter()
            form_retries_used = 0
            pages_failed = 0
            pages_meta = []

            for page_image in form_page_images:
                img_b64 = page_image[2]
                page_height = page_image[3]
                page_width = page_image[4]

                prompt = build_prompt(fp.task_type, fp.prompt_variant, fp.label_set_variant, page_height, page_width)

                page_preds = []
                page_errors = []
                parsed_elements = 0
                validated_elements = 0
                parse_used_salvage = False
                validation_field_failures: dict = {}
                failed_raw_outputs = []
                attempt = 0
                # Retries cover both API errors and parse failures
                for attempt in range(3):
                    raw_output = None
                    try:
                        raw_output = self.call(prompt, img_b64, schema=LayoutAnalysisOutput.model_json_schema())
                    except Exception as e:
                        wait = 2 ** attempt
                        page_errors.append(f"api_error:{type(e).__name__}")
                        print(f"[WARNING] API error for form {form_id} (attempt {attempt+1}/3): {type(e).__name__}: {e}; retrying in {wait}s")
                        time.sleep(wait)
                        continue

                    candidate = []
                    parse_ok = False
                    parsed_n = 0
                    salvage_this_attempt = False
                    rejection_paths_this_attempt: list = []
                    if raw_output:
                        try:
                            parsed, parse_meta = parse_json(raw_output, return_meta=True)
                            parse_ok = True
                            salvage_this_attempt = parse_meta["used_salvage"]
                            parsed = remove_wrapper(parsed)
                            if isinstance(parsed, dict):
                                parsed = [parsed]
                            if isinstance(parsed, list):
                                parsed_n = len(parsed)
                                for element in parsed:
                                    try:
                                        validated = LayoutElement.model_validate(element)
                                        candidate.append(validated.model_dump(by_alias=True))
                                    except Exception as ve:
                                        # Capture Pydantic field paths if available (".".join of loc tuple).
                                        errors_list = getattr(ve, "errors", None)
                                        if callable(errors_list):
                                            try:
                                                for err in ve.errors():
                                                    loc = err.get("loc", ())
                                                    field = ".".join(str(x) for x in loc) or "<root>"
                                                    rejection_paths_this_attempt.append(field)
                                            except Exception:
                                                rejection_paths_this_attempt.append(type(ve).__name__)
                                        else:
                                            rejection_paths_this_attempt.append(type(ve).__name__)
                        except Exception:
                            pass

                    if candidate:
                        page_preds = candidate
                        parsed_elements = parsed_n
                        validated_elements = len(candidate)
                        parse_used_salvage = parse_used_salvage or salvage_this_attempt
                        for f in rejection_paths_this_attempt:
                            validation_field_failures[f] = validation_field_failures.get(f, 0) + 1
                        break

                    if raw_output:
                        failed_raw_outputs.append(raw_output[:8000])  # cap to keep file size sane
                    if not raw_output:
                        page_errors.append("empty_response")
                    elif not parse_ok:
                        page_errors.append("parse_failure")
                    elif parsed_n == 0:
                        page_errors.append("empty_parsed_list")
                    else:
                        page_errors.append("validation_failure_full")
                        for f in rejection_paths_this_attempt:
                            validation_field_failures[f] = validation_field_failures.get(f, 0) + 1
                    parse_used_salvage = parse_used_salvage or salvage_this_attempt
                    if attempt < 2:
                        wait = 2 ** attempt
                        print(f"[WARNING] Empty/unparseable response for form {form_id} page (attempt {attempt+1}/3); retry in {wait}s")
                        time.sleep(wait)

                form_preds["preds"].append(page_preds)
                form_retries_used += attempt
                page_meta = {
                    "retries_used": attempt,
                    "errors": page_errors,
                    "boxes": len(page_preds),
                    "parsed_elements": parsed_elements,
                    "validated_elements": validated_elements,
                    "parse_used_salvage": parse_used_salvage,
                    "validation_field_failures": validation_field_failures,
                }
                if not page_preds:
                    pages_failed += 1
                    page_meta["failed_raw_outputs"] = failed_raw_outputs
                pages_meta.append(page_meta)
                time.sleep(2.0)

            form_preds["meta"] = {
                "latency_seconds": round(time.perf_counter() - t_form_start, 3),
                "retries_used": form_retries_used,
                "pages_failed": pages_failed,
                "pages": pages_meta,
            }
            formatted_predictions.append(form_preds)
            # Persist after each form for crash resilience (atomic via tmp + replace)
            tmp_path = f"{prediction_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(formatted_predictions, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, prediction_path)

            print(f"Completed form {idx}/{total_form_cnt} ({idx/total_form_cnt*100:.2f}%)")

        return formatted_predictions

    def _run_kie(self) -> List[dict]:
        fp = self.file_processor
        annotations = fp.load_annotations()
        prediction_path = fp.get_prediction_path()

        done_ids = set()
        if os.path.exists(prediction_path):
            with open(prediction_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done_ids.add(json.loads(line)["form_id"])
            print(f"Resuming — {len(done_ids)} form(s) already done, skipping.")

        curr_form_cnt, total_form_cnt = 0, len(annotations)
        with open(prediction_path, "a", encoding="utf-8") as f:
            for form in annotations:
                form_id = form["id"]
                if form_id in done_ids:
                    curr_form_cnt += 1
                    continue
                for page_annotations in form["annotations"]:
                    page_idx = page_annotations["page"]
                    image_bytes = fp.get_image(form, page_idx)
                    if image_bytes is None:
                        print(f"[WARNING] Skipping form {form_id} page {page_idx}: image not found.")
                        continue
                    kv_pairs = page_annotations["annotations"]

                    curr_page_details = {"form_id": form_id, "page": page_idx, "kie_details": []}

                    for kv_pair in kv_pairs:
                        key = kv_pair.get("key_text")
                        value = kv_pair.get("value_name")
                        if not key or not value:
                            continue

                        question = f"What is the value of {key}?"
                        prompt = build_prompt(fp.task_type, fp.prompt_variant, question=question)
                        try:
                            output = self.call(prompt, image_bytes)
                        except Exception as e:
                            safe_key = key.encode('ascii', errors='replace').decode('ascii')
                            print(f"[WARNING] API error for key '{safe_key}' (form {form_id}): {type(e).__name__}")
                            output = None

                        if not output:
                            pred = "PARSING_ERROR"
                        else:
                            output = output[:4000]  # cap to avoid memory issues on unexpectedly large responses
                            match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
                            pred = match.group(1).strip() if match else "PARSING_ERROR"

                        curr_page_details["kie_details"].append({
                            "key": key,
                            "question": question,
                            "pred_val": pred
                        })

                        time.sleep(1.0)  # avoid 503 on sequential calls

                    f.write(json.dumps(curr_page_details, ensure_ascii=False) + "\n")

                curr_form_cnt += 1
                print(f"Completed form {curr_form_cnt}/{total_form_cnt} ({curr_form_cnt/total_form_cnt*100:.2f}%)")


class BatchInferencer(Inferencer):
    """Batch API inference: covers DLA with OpenAI / Gemini / Claude batch APIs."""

    def __init__(self, file_processor: FileProcessor, model_config: ModelConfig, inference_config: InferenceConfig):
        super().__init__(file_processor, model_config, inference_config)
        self.model_path = NATIVE_MODEL_PATH_MAP[model_config.model_family][file_processor.model_name]
        self.openai_client = OpenAI()
        self.gemini_client = genai.Client()
        self.claude_client = Anthropic()

    def _build_request_body(self, prompt: str, img_b64: str) -> dict:
        body = {}
        if self.model_family == "openai":
            body = {
                "model": self.model_path,
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                ]}],
            }
        elif self.model_family == "gemini":
            body = {
                "contents": [{"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": {"data": img_b64, "mime_type": "image/jpeg"}},
                ]}]
            }
        elif self.model_family == "claude":
            body = {
                "model": self.model_path,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"data": img_b64, "media_type": "image/jpeg", "type": "base64"}},
                ]}],
            }

        if self.thinking_mode == "off":
            reasoning_effort, max_output_tokens, max_reasoning_tokens = "low", 16000, 1024
        else:
            reasoning_effort, max_output_tokens, max_reasoning_tokens = "high", 64000, 48000

        if self.model_family == "claude":
            body["temperature"] = self.temperature
            body["max_tokens"] = max_output_tokens
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": reasoning_effort}
        elif self.model_family == "openai":
            body["temperature"] = self.temperature
            body["max_output_tokens"] = max_output_tokens
            body["reasoning"] = {"effort": reasoning_effort}
        elif self.model_family == "gemini":
            body["generation_config"] = {
                "thinking_config": {"thinking_budget": max_reasoning_tokens},
                "temperature": self.temperature,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "application/json",
            }
        return body

    def _build_request(self, inference_id: str, body: dict) -> dict:
        if self.model_family == "openai":
            return {"custom_id": inference_id, "method": "POST", "url": "/v1/responses", "body": body}
        elif self.model_family == "gemini":
            return {"key": inference_id, "request": body}
        elif self.model_family == "claude":
            return {"custom_id": inference_id, "params": body}

    def _upload_batch_file(self, batch_file_path: str) -> dict:
        if self.model_family == "openai":
            batch_file = self.openai_client.files.create(file=open(batch_file_path, "rb"), purpose="batch")
            return dict(id=batch_file.id, filename=batch_file.filename, purpose=batch_file.purpose, created_at=batch_file.created_at)
        elif self.model_family == "gemini":
            batch_file = self.gemini_client.files.upload(
                file=batch_file_path,
                config=types.UploadFileConfig(display_name=batch_file_path.split("/")[1], mime_type="jsonl")
            )
            created_at = batch_file.create_time
            if isinstance(created_at, datetime):
                created_at = int(created_at.replace(tzinfo=created_at.tzinfo or timezone.utc).timestamp())
            return dict(id=batch_file.name, filename=batch_file.display_name, purpose="batch", created_at=created_at)

    def _create_batch_job(self, input_file_id: str, claude_requests=None) -> dict:
        if self.model_family == "openai":
            batch = self.openai_client.batches.create(input_file_id=input_file_id, endpoint="/v1/responses", completion_window="24h")
            return dict(batch_id=batch.id, batch_status=batch.status, batch_created_at=batch.created_at)
        elif self.model_family == "gemini":
            batch = self.gemini_client.batches.create(model=self.model_path, src=input_file_id)
            created_at = batch.create_time
            if isinstance(created_at, datetime):
                created_at = int(created_at.timestamp())
            return dict(batch_id=batch.name, batch_status=batch.state.value, batch_created_at=created_at)
        elif self.model_family == "claude":
            batch = self.claude_client.messages.batches.create(requests=claude_requests)
            created_at = batch.created_at
            if isinstance(created_at, datetime):
                created_at = int(created_at.timestamp())
            return dict(batch_id=batch.id, batch_status=batch.processing_status, batch_created_at=created_at)

    def submit(self) -> dict:
        """Build batch input file, upload, create job. Returns combined file+batch metadata."""
        fp = self.file_processor
        annotations = fp.load_annotations()
        claude_requests = []

        with open(fp.batch_input_file_path, "w", encoding="utf-8") as f:
            for form_annotations in tqdm(annotations, desc=f"Building {fp.task_type.upper()} batch"):
                form_id = form_annotations["id"]
                form_page_images = load_form_pages(fp.subset_dir, form_annotations)

                for page_idx, page_image in enumerate(form_page_images):
                    inference_id = f"{form_id}_page_{page_idx}"
                    img_b64, page_height, page_width = page_image[2], page_image[3], page_image[4]

                    prompt = build_prompt(fp.task_type, fp.prompt_variant, fp.label_set_variant, page_height, page_width)
                    body = self._build_request_body(prompt, img_b64)
                    record = self._build_request(inference_id, body)

                    if self.model_family == "claude":
                        claude_requests.append(record)
                    else:
                        f.write(json.dumps(record) + "\n")

        file_details = self._upload_batch_file(fp.batch_input_file_path)
        batch_details = self._create_batch_job(
            input_file_id=file_details["id"],
            claude_requests=claude_requests if self.model_family == "claude" else None,
        )
        return {**file_details, **batch_details}

    def poll(self, batch_id: str) -> str:
        """Check batch job status, return status string."""
        raise NotImplementedError

    def retrieve(self, batch_id: str, prediction_path: str, error_path: str = None):
        """Retrieve completed batch results and write to prediction_path."""
        if self.model_family == "claude":
            batch = self.claude_client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                with open(prediction_path, "w", encoding="utf-8") as f:
                    for item in self.claude_client.messages.batches.results(batch_id):
                        f.write(json.dumps(item.model_dump() if hasattr(item, "model_dump") else item, ensure_ascii=False) + "\n")
        elif self.model_family == "openai":
            batch = self.openai_client.batches.retrieve(batch_id)
            if batch.status in {"completed", "failed"}:
                if batch.output_file_id:
                    Path(prediction_path).write_bytes(self.openai_client.files.content(batch.output_file_id).read())
                if batch.error_file_id and error_path:
                    Path(error_path).write_bytes(self.openai_client.files.content(batch.error_file_id).read())
        elif self.model_family == "gemini":
            batch = self.gemini_client.batches.get(name=batch_id)
            if batch.state.name in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}:
                if batch.dest and batch.dest.file_name:
                    Path(prediction_path).write_bytes(self.gemini_client.files.download(file=batch.dest.file_name))

    def run(self) -> dict:
        """Submit batch job. Retrieve is triggered separately via postprocess step."""
        return self.submit()


class TaskRunner:
    def __init__(self, file_processor: FileProcessor, inferencer: Inferencer):
        self.file_processor = file_processor
        self.inferencer = inferencer

    def run(self):
        return self.inferencer.run()
