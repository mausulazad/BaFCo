import argparse
import glob
import json

from dotenv import load_dotenv
load_dotenv()

from postprocess_blocks import BatchPostprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Form Layout Detection Benchmarking")

    parser.add_argument(
        "--model_family", required=True, type=str, help="Model Family (e.g. qwen, openai)"
    )
    parser.add_argument(
        "--model_name", required=True, type=str, help="Model Name (e.g. qwen_3.6, gemini_3)"
    )
    parser.add_argument(
        "--task_type", required=True, type=str.lower, choices=["dla", "kie"], help="Task type (dla, kie)"
    )
    parser.add_argument(
        "--thinking_mode", required=True, type=str.lower, choices=["on", "off"],
        help="on for reasoning models, off for non-reasoning ones",
    )
    parser.add_argument(
        "--prompt_variant", required=True, type=str.lower, choices=["zero_shot", "cot"],
        help="Prompt variant (zero_shot, cot; KIE supports zero_shot only)",
    )
    parser.add_argument(
        "--label_set_variant", required=True, type=str.lower, choices=["full", "reduced"],
        help="Label set (full = 26 entities, reduced = 5 coarse; DLA only)",
    )
    parser.add_argument("--dataset", default="bafco", type=str, help="Dataset name (output-path namespace)")
    parser.add_argument(
        "--release_root", required=True, type=str,
        help="Root of the downloaded BaFCo export (contains bafco_dla/ and bafco_kie/).",
    )
    parser.add_argument("--lang", required=False, type=str.lower, default="all", choices=["en", "bn", "all"], help="Language tag for output paths (en, bn, all)")
    parser.add_argument(
        "--batch_size", required=True, type=int, help="Batch size"
    )
    parser.add_argument(
        "--batch_num", required=True, type=int, help="Batch Number"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_family = args.model_family.strip().lower()
    model_name = args.model_name.strip().lower()
    task_type = args.task_type.strip().lower()
    thinking_mode = args.thinking_mode.strip().lower()
    prompt_variant = args.prompt_variant.strip().lower()
    label_set_variant = args.label_set_variant.strip().lower()
    dataset = args.dataset.strip().lower()
    release_root = args.release_root
    lang = args.lang.strip().lower()
    batch_size = args.batch_size
    batch_num = args.batch_num

    if task_type == "kie":
        reasoning_state = "nonreasoning" if thinking_mode == "off" else "reasoning"
        pattern = f"predictions/{dataset}/kie/{lang}/{model_name}_{prompt_variant}_{reasoning_state}_*.jsonl"
        matches = glob.glob(pattern)
        if not matches:
            print(f"No KIE prediction file found matching: {pattern}")
            return
        prediction_file_path = matches[0]
        rows = []
        with open(prediction_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        total_kv = sum(len(row.get("kie_details", [])) for row in rows)
        print(f"KIE prediction file: {prediction_file_path}")
        print(f"Forms: {len(rows)}, KV pairs: {total_kv}")
        return

    pp = BatchPostprocessor(
        model_family, task_type, label_set_variant,
        model_name, release_root, dataset, prompt_variant, thinking_mode,
        lang, batch_size, batch_num,
    )

    with open(pp.metadata_file_path, "r") as f:
        batch_metadata = json.load(f)

    batch_id = batch_metadata["batch_id"]
    batch_details = pp.fetch_batch_status(batch_id)

    if not (pp.prediction_file_path.exists() or pp.error_file_path.exists()):
        pp.retrieve_output(batch_details, pp.prediction_file_path, pp.error_file_path)

    raw_preds = []
    if pp.prediction_file_path.exists():
        with pp.prediction_file_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw_preds.append(json.loads(line))

    sorted_preds = pp.sort_predictions(raw_preds)
    standardized_predictions = pp.standardize_predictions(sorted_preds)
    parsed_predictions, parsed_cnt = pp.parse_predictions(standardized_predictions)
    print(f"Total Parsed Predictions: {parsed_cnt}")
    finalized_predictions, valid_cnt, invalid_cnt = pp.validate_predictions(parsed_predictions)
    print(f"Total Valid (Structurally) Predictions: {valid_cnt}")
    print(f"Total Invalid (Structurally) Predictions: {invalid_cnt}")

    formatted_predictions = pp.group_by_form(finalized_predictions)

    with open(pp.output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_predictions, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()