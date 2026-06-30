import argparse

from dotenv import load_dotenv

from inference_blocks import ModelConfig, InferenceConfig, FileProcessor, SyncInferencer, BatchInferencer, TaskRunner
from utils import NATIVE_MODEL_PATH_MAP

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Form Layout Detection Benchmarking")

    parser.add_argument(
        "--model_family", required=True, type=str, help="Model Family (e.g. qwen, openai)"
    )
    parser.add_argument(
        "--model_name", required=True, type=str, help="Model Name (e.g. qwen_3.6, gemini_3)"
    )
    parser.add_argument(
        "--temperature", required=True, type=float
    )
    parser.add_argument(
        "--top_p", required=True, type=float
    )
    parser.add_argument(
        "--thinking_mode", required=True, type=str.lower, choices=["on", "off"],
        help="on for reasoning models, off for non-reasoning ones",
    )
    parser.add_argument(
        "--batch_size", required=True, type=int, help="Batch size"
    )
    parser.add_argument(
        "--batch_num", required=True, type=int, help="Batch Number"
    )
    parser.add_argument(
        "--tuning_free", action="store_true"
    )
    parser.add_argument(
        "--use_api", action="store_true"
    )
    parser.add_argument(
        "--use_openrouter", action=argparse.BooleanOptionalAction, default=True,
        help="Route sequential inference through OpenRouter (default). Pass --no-use_openrouter for native clients."
    )
    parser.add_argument(
        "--prompt_variant", required=True, type=str.lower, choices=["zero_shot", "cot"],
        help="Prompt variant (zero_shot, cot; KIE supports zero_shot only)",
    )
    parser.add_argument(
        "--label_set_variant", required=True, type=str.lower, choices=["full", "reduced"],
        help="Label set (full = 26 entities, reduced = 5 coarse; DLA only)",
    )
    parser.add_argument(
        "--task_type", required=True, type=str.lower, choices=["dla", "kie"], help="Task type (dla, kie)"
    )
    parser.add_argument("--dataset", default="bafco", type=str, help="Dataset name (output-path namespace)")
    parser.add_argument(
        "--release_root", required=True, type=str,
        help="Root of the downloaded BaFCo export (contains bafco_dla/ and bafco_kie/).",
    )
    parser.add_argument("--lang", default="all", type=str.lower, choices=["en", "bn", "all"], help="Language tag for output paths (en, bn, all)")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_family = args.model_family.strip().lower()
    model_name = args.model_name.strip().lower()
    temperature = args.temperature
    top_p = args.top_p
    thinking_mode = args.thinking_mode.strip().lower()
    batch_size = args.batch_size
    batch_num = args.batch_num
    tuning_free = args.tuning_free
    use_api = args.use_api
    prompt_variant = args.prompt_variant.strip().lower()
    label_set_variant = args.label_set_variant.strip().lower()
    task_type = args.task_type.strip().lower()
    dataset = args.dataset.strip().lower()
    release_root = args.release_root
    lang = args.lang.strip().lower()

    if task_type not in ("dla", "kie"):
        raise ValueError(f"Unknown task_type: {task_type}. Expected 'dla' or 'kie'.")

    file_processor = FileProcessor(
        task_type, model_name, release_root,
        batch_size, batch_num, prompt_variant, label_set_variant, thinking_mode,
        lang=lang, dataset_name=dataset,
    )
    model_config = ModelConfig(tuning_free, model_family)
    inference_config = InferenceConfig(use_api, temperature, top_p, thinking_mode, args.use_openrouter)

    native_batch_supported = (
        model_family in NATIVE_MODEL_PATH_MAP
        and model_name in NATIVE_MODEL_PATH_MAP[model_family]
    )

    if use_api and task_type == "dla" and native_batch_supported:
        inferencer = BatchInferencer(file_processor, model_config, inference_config)
    else:
        if use_api and task_type == "dla" and not native_batch_supported:
            print(f"⚠️ {model_family}/{model_name} not in NATIVE_MODEL_PATH_MAP — falling back to SyncInferencer (OpenRouter routing).")
        inferencer = SyncInferencer(file_processor, model_config, inference_config)

    runner = TaskRunner(file_processor, inferencer)
    runner.run()


if __name__ == "__main__":
    main()