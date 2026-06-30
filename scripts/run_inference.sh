#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$BASE_DIR/benchmark"

# Path to the downloaded BaFCo dataset export (contains bafco_dla/ and bafco_kie/).
release_root="${RELEASE_ROOT:-$BASE_DIR/bafco_data}"

task_type="dla"              # dla, kie
prompt_variant="zero_shot"   # zero_shot, cot
label_set_variant="full"     # DLA only: "reduced" (5 labels) / "full" (26 labels)
model_family="qwen"          # gemini, openai, qwen, claude, kimi
model_name="qwen_3.6"
temperature=1.0
top_p=1.0
thinking_mode="off"          # on/off
batch_size=50                # forms per batch
batch_num=1                  # which batch to run; bump between invocations to cover the set

echo "===== ${task_type} inference, batch ${batch_num} (size=${batch_size}) ====="
PY="${PYTHON:-$BASE_DIR/.venv/Scripts/python.exe}"
PYTHONIOENCODING=utf-8 "$PY" inference.py \
  --task_type "$task_type" \
  --release_root "$release_root" \
  --model_family "$model_family" \
  --model_name "$model_name" \
  --temperature "$temperature" \
  --top_p "$top_p" \
  --thinking_mode "$thinking_mode" \
  --batch_size "$batch_size" \
  --batch_num "$batch_num" \
  --tuning_free \
  --prompt_variant "$prompt_variant" \
  --label_set_variant "$label_set_variant"
echo "===== Completed batch ${batch_num} ====="
