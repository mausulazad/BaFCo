# BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension

## ECCV 2026

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2607.05614) [![Dataset](https://img.shields.io/badge/Dataset-555?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/datasets/Mausul/bafco)

> 📋 **TL;DR:** BaFCo is the first benchmark for **Bangla form comprehension**:
> 200 complex multi-page government forms (316 pages, 15 domains) annotated with
> **16,382 layout entities and 8,771 inter-field relationships** under a fine-grained
> 26-entity schema (+ a 5-type coarse set) for Document Layout Analysis, plus
> **1,926 key-value pairs** across 156 forms for Key Information Extraction. Even
> flagship MLLMs (GPT, Gemini, Claude, Qwen, Kimi) struggle, especially at
> localizing fine-grained entities during DLA.

_**Abstract:** Document comprehension is a challenging yet impactful task for
Multimodal Large Language Models, especially as these systems see growing adoption
in real-world, human-centric applications. However, this adoption is limited for
low-resource languages such as Bangla due to the scarcity of high-quality annotated
data. To address this gap, we introduce BaFCo, a benchmark dataset for Bangla form
comprehension with a focus on Document Layout Analysis (DLA) and Key Information
Extraction (KIE). BaFCo curates 200 multi-page complex Bangladeshi government forms,
sourced from across diverse sectors including agriculture, education, banking, and
land management. To accurately capture the structural and contextual complexity of
these forms, we define a fine-grained annotation schema comprising 26 types of form
entities, along with a separate coarse form entity set consisting of 5 types. We
evaluate the latest MLLMs from the ChatGPT, Gemini, Claude, Qwen, and Kimi series
using zero-shot and chain-of-thought prompts under both low and high reasoning
setups. Our results reveal limitations in current MLLMs' ability in comprehending
Bangla forms, particularly in accurately localizing highly granular form entities._

This repository provides the inference and evaluation pipeline for both tasks, DLA and KIE.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your OpenRouter API key
```

Inference routes through [OpenRouter](https://openrouter.ai) by default, so only
`OPEN_ROUTER_API_KEY` is required. Native-SDK and batch-API paths additionally
use `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `ANTHROPIC_API_KEY`.

## Dataset

```python
from huggingface_hub import snapshot_download
snapshot_download("Mausul/bafco", repo_type="dataset", local_dir="bafco_data")
```

Point the pipeline at the root with `--release_root bafco_data`; the task is
selected by `--task_type`. The export is laid out as:

```
bafco_data/
├── bafco_dla/                  #   Document Layout Analysis (DLA)
│   ├── all_annotations.json    #   per-form pages, bounding boxes, and relations
│   ├── images/                 #   pages nested by domain → form: <domain>/form_<id>__<slug>/<name>_page_<n>.jpg
│   └── README.md
└── bafco_kie/                  #   Key Information Extraction (KIE)
    ├── all_annotations.json    #   per-form, per-page key–value pairs
    ├── bn_annotations.json     #   Bangla splits of all_annotations.json
    ├── en_annotations.json     #   English splits of all_annotations.json
    ├── bn/                     #   Bangla form pages:  form_<id>/<name>_page_<n>.jpg
    ├── en/                     #   English form pages: form_<id>/<name>_page_<n>.jpg
    └── README.md
```

The pipeline reads `all_annotations.json` from each subset; the `bn`/`en`
annotation files for KIE are splits provided for convenient language specific analysis.

## Run

Inference and evaluation take the same `--task_type` / `--release_root` pair for
both tasks. From `benchmark/`:

```bash
# DLA Inference
python inference.py --task_type dla --release_root ../bafco_data \
  --model_family qwen --model_name qwen_3.6 \
  --prompt_variant zero_shot --label_set_variant full \
  --thinking_mode off --temperature 1.0 --top_p 1.0 \
  --batch_size 50 --batch_num 1 --tuning_free

# DLA Evaluation
python eval.py --task_type dla --release_root ../bafco_data \
  --model_name qwen_3.6 --prompt_variant zero_shot \
  --label_set_variant full --granularity_level high --thinking_mode off --lang all

# KIE Inference
python inference.py --task_type kie --release_root ../bafco_data \
  --model_family qwen --model_name qwen_3.6 \
  --prompt_variant zero_shot --label_set_variant full \
  --thinking_mode off --temperature 1.0 --top_p 1.0 \
  --batch_size 50 --batch_num 1 --tuning_free

# KIE Evaluation
python eval.py --task_type kie --release_root ../bafco_data \
  --model_name qwen_3.6 --prompt_variant zero_shot --thinking_mode off --lang all
```

Alternatively, `scripts/` has a wrapper for each step — `run_inference.sh`,
`run_eval.sh`, and `run_postprocessing.sh` (batch-API path). Edit the variables
at the top (`RELEASE_ROOT`, model, task) and run.

Accepted values for the main options (run `--help` for the full list):

| Argument | Values |
| --- | --- |
| `--task_type` | `dla`, `kie` |
| `--model_family` | `openai`, `gemini`, `claude`, `qwen`, `kimi` |
| `--model_name` | `gpt_5.2`, `gemini_3`, `claude_opus_4.6`, `qwen_3.6`, `kimi_k2.5` |
| `--prompt_variant` | `zero_shot`, `cot` (KIE: `zero_shot` only) |
| `--label_set_variant` | `full` (26 entities), `reduced` (5 coarse) (DLA only) |
| `--granularity_level` | `high` (eval on 26), `low` (eval on 5) (DLA only) |
| `--thinking_mode` | `on`, `off` |
| `--lang` | `en`, `bn`, `all` |

## License

Code: MIT (see `LICENSE`), Dataset: CC-BY-NC-4.0 (on Hugging Face)

## Citation

```bibtex
@misc{azad2026bafcodocumentunderstandingbenchmark,
      title={BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension}, 
      author={Abu Tyeb Azad and Ishita Sur Apan and Fahim Ahmed and Sumaiya Karim Katha and Ezharuddin Jubaer and Armun Alam and Pranjal Kumar Nandi and Amin Ahsan Ali and Aman Chadha and Md Mofijul Islam and AKM Mahbubur Rahman},
      year={2026},
      eprint={2607.05614},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.05614}, 
}
```
