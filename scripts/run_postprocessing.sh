#!/usr/bin/env bash
set -euo pipefail

# Normalizes batch-API results into prediction files (batch path only; the default sync path needs no postprocessing).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$BASE_DIR/benchmark"

# Path to the downloaded BaFCo dataset export (contains bafco_dla/ and bafco_kie/).
release_root="${RELEASE_ROOT:-$BASE_DIR/bafco_data}"

task_type="dla"              # dla, kie
prompt_variant="zero_shot"   # zero_shot, cot
label_set_variant="full"     # "reduced" (5 labels) / "full" (26 labels)
model_family="openai"        # gemini, openai, claude
model_name="gpt_5.2"
thinking_mode="off"          # on/off
batch_size=50
batch_num=1
lang="all"                   # en, bn, all

PY="${PYTHON:-$BASE_DIR/.venv/Scripts/python.exe}"
PYTHONIOENCODING=utf-8 "$PY" postprocess.py \
  --task_type "$task_type" \
  --release_root "$release_root" \
  --model_family "$model_family" \
  --model_name "$model_name" \
  --thinking_mode "$thinking_mode" \
  --prompt_variant "$prompt_variant" \
  --label_set_variant "$label_set_variant" \
  --batch_size "$batch_size" \
  --batch_num "$batch_num" \
  --lang "$lang"
