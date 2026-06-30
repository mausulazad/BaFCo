#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$BASE_DIR/benchmark"

# Path to the downloaded BaFCo dataset export (contains bafco_dla/ and bafco_kie/).
release_root="${RELEASE_ROOT:-$BASE_DIR/bafco_data}"

task_type="dla"                   # dla, kie
prompt_variant="zero_shot"        # zero_shot, cot
label_set_variant="full"          # DLA only: predictions label set ("reduced"/"full")
granularity_level="high"          # DLA only: eval label count ("low" = 5, "high" = 26)
model_name="qwen_3.6"
thinking_mode="off"               # on/off
lang="all"                        # en, bn, all (KIE filters by dataset language)
no_visualize=true                 # true: skip per-page bbox-overlay PNGs (DLA); metrics still computed

PY="${PYTHON:-$BASE_DIR/.venv/Scripts/python.exe}"
PYTHONIOENCODING=utf-8 "$PY" eval.py \
  --task_type "$task_type" \
  --release_root "$release_root" \
  --model_name "$model_name" \
  --thinking_mode "$thinking_mode" \
  --prompt_variant "$prompt_variant" \
  --label_set_variant "$label_set_variant" \
  --granularity_level "$granularity_level" \
  --lang "$lang" \
  $( [ "$no_visualize" = "true" ] && echo "--no_visualize" )
