import argparse
import csv
import json
import os
import glob
import re
import random

from PIL import Image, ImageDraw, ImageFont

from utils import (
    format_annotations,
    build_gt_index,
    build_pred_index,
    filter_indices_by_forms,
    preprocess_labels,
    compute_map_at_iou,
    rescale_predictions_to_gt,
)

from kie_utils import get_kie_eval
from release_loader import load_dla_targets, load_kie_forms, subset_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Form Layout Detection Benchmarking")

    parser.add_argument("--dataset", default="bafco", type=str, help="Dataset name (output-path namespace)")
    parser.add_argument(
        "--release_root", required=True, type=str,
        help="Root of the downloaded BaFCo export (contains bafco_dla/ and bafco_kie/).",
    )
    parser.add_argument(
        "--task_type", required=True, type=str.lower, choices=["dla", "kie"], help="Task type (dla, kie)"
    )
    parser.add_argument(
        "--model_name", required=True, type=str, help="Model Name (e.g. qwen_3.6, gemini_3)"
    )
    parser.add_argument(
        "--prompt_variant", required=True, type=str.lower, choices=["zero_shot", "cot"],
        help="Prompt variant (zero_shot, cot; KIE supports zero_shot only)",
    )
    parser.add_argument(
        "--label_set_variant", default="full", type=str.lower, choices=["full", "reduced"],
        help="DLA only: label set (full = 26 entities, reduced = 5 coarse)",
    )
    parser.add_argument(
        "--thinking_mode", required=True, type=str.lower, choices=["on", "off"], help="Reasoning mode: on/off"
    )
    parser.add_argument(
        "--granularity_level", default="low", type=str.lower, choices=["low", "high"],
        help="DLA only: eval label count (low = 5, high = 26)",
    )
    parser.add_argument("--lang", required=False, type=str.lower, default="all", choices=["en", "bn", "all"], help="Language filter (en/bn/all)")
    parser.add_argument("--batch_size", required=False, type=int, default=50, help="Batch size used during inference")
    parser.add_argument("--batch_num", required=False, type=int, default=1, help="Batch number used during inference")
    parser.add_argument(
        "--difficulty_sheet", required=False, type=str, default=None,
        help="Optional CSV with columns form_id,difficulty (easy/medium/hard) for bucketed mAP",
    )
    parser.add_argument(
        "--no_visualize", action="store_true",
        help="Skip writing per-page bbox-overlay PNGs (saves disk space; metrics still computed)",
    )
    args = parser.parse_args()

    return args


DIFFICULTY_BUCKETS = ("easy", "medium", "hard")


def load_difficulty_map(path: str) -> dict:
    """Load form_id -> bucket from a CSV with columns form_id,difficulty.

    Rejects unknown bucket values and missing columns with a clear error.
    Uses utf-8-sig to tolerate a BOM from Excel exports.
    """
    allowed = set(DIFFICULTY_BUCKETS)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "form_id" not in fields or "difficulty" not in fields:
            raise ValueError(
                f"Difficulty sheet must have columns form_id,difficulty; got {fields}"
            )
        mapping: dict = {}
        for row in reader:
            raw_id = (row.get("form_id") or "").strip()
            if not raw_id:
                continue
            fid = int(raw_id)
            bucket = (row.get("difficulty") or "").strip().lower()
            if bucket not in allowed:
                raise ValueError(
                    f"form_id={fid}: unknown difficulty '{bucket}' (expected one of {sorted(allowed)})"
                )
            mapping[fid] = bucket
    return mapping


def resolve_image_path(page_path: str, source_images_dir: str) -> str:
    """Normalize stored page paths and join with source_images_dir."""
    if page_path.startswith("/data/local-files/?d=/forms_jpg"):
        page_path = page_path.replace("/data/local-files/?d=/forms_jpg", "")
    page_path = page_path.lstrip("/")
    return os.path.join(source_images_dir, page_path)

def get_scaled_font(image_width, base=1000, min_size=16, max_size=48):
    size = int((image_width / base) * 24)
    size = max(min_size, min(size, max_size))
    for candidate in ("DejaVuSans-Bold.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()

def draw_boxes_on_image(image_path, pred_boxes, target_boxes, save_path, label_set_variant):
    """Draw predicted and target boxes on the image with labels."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # scale font to image width
    if target_boxes:
        image_width = target_boxes[0]["original_width"]
        font = get_scaled_font(image_width)
    else:
        font = ImageFont.load_default()

    def draw_box(box, label, outline, text_bg):
        x1, y1, x2, y2 = box
        
        # reorder when needed
        if x2 < x1 or y2 < y1:
            print("Found a reordered bounding box:", (x1, y1, x2, y2))
        
        if x2 < x1:
            x1, x2 = x2, x1

        if y2 < y1:
            y1, y2 = y2, y1

        draw.rectangle([x1, y1, x2, y2], outline=outline, width=3)
        if label:
            padding = 4
            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]

            # Draw label inside box, if near the top edge of the page
            text_y = y1 - th - padding * 2
            if text_y < 0:
                text_y = y1 + padding

            draw.rectangle(
                [x1, text_y, x1 + tw + padding * 2, text_y + th + padding * 2],
                fill=text_bg
            )
            draw.text(
                (x1 + padding, text_y + padding),
                label,
                fill="black",
                font=font
            )

    # targets in green
    for t in target_boxes:
        coords = t["denormalized_coordinates"]
        x1 = coords["x"]
        y1 = coords["y"]
        x2 = x1 + coords["width"]
        y2 = y1 + coords["height"]
        label = t.get("label", "")
        if label_set_variant == "reduced":
            if label in { "Header", "Title of Form", "Footer", "Section Title" }:
                label = "Headings"
            elif label in { "Table" }:
                label = "Table"
            elif label in { "Figure/Diagram/Logo" }:
                label = "Image"
            elif label in { "Form Key", "Form Value", "Inline Key", "Inline Value", "Signature Key", "Signature Val", "Tick Mark", "Checkbox", "Photo Field" }:
                label = "Fields"
            elif label in { "Text Block", "Page Num", "Gibberish, Mark for removal", "Others" }:
                label = "Others"

            if label in { "Headings", "Table", "Image", "Fields", "Others" }:
                draw_box((x1, y1, x2, y2), f"T:{label}", outline="lime", text_bg="white")
            else:
                continue
        elif label_set_variant == "full":
            draw_box((x1, y1, x2, y2), f"T:{label}", outline="lime", text_bg="white")
    
    # predictions in red
    for p in pred_boxes:
        x, y, w, h = p["bbox"]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        label = p.get("class", "")
        draw_box((x1, y1, x2, y2), f"P:{label}", outline="red", text_bg="yellow")

    image.save(save_path)


def visualize(task_type, thinking_mode, label_set_variant, predictions, target_annotations, targets, dataset, model_name, prompt_variant, source_images_dir):
    """Render predicted and target boxes on the original pages and save images."""
    reasoning_effort = "high" if thinking_mode == "on" else "low"
    vis_dir = f"evaluations/{dataset}/visualizations/{task_type}/{label_set_variant}/{model_name}/{reasoning_effort}_reasoning_{prompt_variant}"
    os.makedirs(vis_dir, exist_ok=True)
    id_to_pages = {form["id"]: form["data"]["pages"] for form in targets}

    paired = list(zip(predictions, target_annotations))
    random.shuffle(paired)

    drawn = 0
    for pred_entry, target_entry in paired:
        if drawn >= 10:
            break
        form_id = pred_entry["form_id"]
        pages = id_to_pages.get(form_id, [])

        # pick single and non-empty pages 
        if len(pages) > 1 or len(target_entry["sorted_bbox_details"]) == 0:
            continue

        img_path = resolve_image_path(pages[0], source_images_dir)
        if not os.path.exists(img_path):
            print(f"Image not found for form {form_id}: {img_path}")
            continue
        out_path = os.path.join(
            vis_dir, f"form_{form_id}.png"
        )

        pred_page = pred_entry["sorted_preds"][0]
        target_page = target_entry["sorted_bbox_details"][0]

        draw_boxes_on_image(img_path, pred_page, target_page, out_path, label_set_variant)
        drawn += 1


def run_kie_eval(model_name, dataset, lang, prompt_variant, reasoning_state, release_root):
    pattern = f"predictions/{dataset}/kie/all/{model_name}_{prompt_variant}_{reasoning_state}_*.jsonl"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No prediction file found matching: {pattern}")
    print(f"Found {len(matches)} prediction file(s): {sorted(matches)}")

    rows = []
    for prediction_path in matches:
        with open(prediction_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    gold_by_page, lang_by_form = {}, {}
    for form in load_kie_forms(os.path.join(subset_dir(release_root, "kie"), "all_annotations.json")):
        lang_by_form[form["id"]] = (form.get("language") or "").strip().lower()
        for page in form["annotations"]:
            gold_by_page[(form["id"], page["page"])] = [
                kv["value_name"].strip()
                for kv in page["annotations"]
                if kv.get("key_text") and kv.get("value_name")
            ]

    if lang in ("en", "bn"):
        rows = [row for row in rows if lang_by_form.get(row["form_id"]) == lang]

    predictions, targets = [], []
    for row in rows:
        preds = [d["pred_val"] for d in row["kie_details"]]
        golds = gold_by_page.get((row["form_id"], row["page"]), [])
        if len(preds) != len(golds):
            print(f"[WARNING] form {row['form_id']} page {row['page']}: "
                  f"{len(preds)} predictions vs {len(golds)} gold values; pairing the first {min(len(preds), len(golds))}.")
        for pred_val, gold_val in zip(preds, golds):
            predictions.append(pred_val)
            targets.append(gold_val)

    return get_kie_eval(predictions, targets)


def load_sorted_targets(dataset, lang, model_name, targets):
    os.makedirs(f"evaluations/{dataset}/sorted_targets/dla/{lang}/{model_name}", exist_ok=True)
    formatted_target_path = f"evaluations/{dataset}/sorted_targets/dla/{lang}/annotations.json"

    if os.path.exists(formatted_target_path):
        with open(formatted_target_path, "r", encoding="utf-8") as f:
            return json.load(f)

    target_annotations = format_annotations(targets)
    sorted_targets = []
    for annotation in target_annotations:
        annotation["sorted_bbox_details"] = []
        for page_bboxes in annotation["bbox_details"]:
            sorted_page = sorted(
                page_bboxes,
                key=lambda bbox: (
                    bbox["denormalized_coordinates"]["y"],
                    bbox["denormalized_coordinates"]["x"],
                ),
            )
            annotation["sorted_bbox_details"].append(sorted_page)
        sorted_annotation = {k: v for k, v in annotation.items() if k != "bbox_details"}
        sorted_targets.append(sorted_annotation)

    with open(formatted_target_path, "w", encoding="utf-8") as f:
        json.dump(sorted_targets, f, indent=2, ensure_ascii=False)
    return sorted_targets


def load_sorted_predictions(dataset, lang, model_name, task_type,
                             prompt_variant, label_set_variant, reasoning_state,
                             combined_predictions):
    os.makedirs(f"evaluations/{dataset}/sorted_prediction/dla/{lang}/{model_name}", exist_ok=True)
    formatted_prediction_path = (
        f"evaluations/{dataset}/sorted_prediction/dla/{lang}/{model_name}/"
        f"{task_type}_annotations_{prompt_variant}_{label_set_variant}_labelset_{reasoning_state}_predictions.json"
    )

    if os.path.exists(formatted_prediction_path):
        with open(formatted_prediction_path, "r", encoding="utf-8") as f:
            return json.load(f)

    sorted_predictions = []
    for form_preds in combined_predictions:
        form_preds["sorted_preds"] = []
        for page_preds in form_preds["preds"]:
            if page_preds is None:
                items = []
            elif isinstance(page_preds, dict):
                items = page_preds.get("fields") or []
            else:
                items = page_preds

            if isinstance(items, list) and len(items) == 1 and items[0] == "failed":
                continue

            sorted_page = sorted(items, key=lambda bbox: (bbox["bbox"][1], bbox["bbox"][0]))
            form_preds["sorted_preds"].append(sorted_page)
        sorted_annotation = {k: v for k, v in form_preds.items() if k != "preds"}
        sorted_predictions.append(sorted_annotation)

    with open(formatted_prediction_path, "w", encoding="utf-8") as f:
        json.dump(sorted_predictions, f, indent=2, ensure_ascii=False)
    return sorted_predictions


def extract_start_idx(path, task_type):
    pattern = rf'(?:^|[/\\]){re.escape(task_type)}_(\d+)_(\d+)_'
    m = re.search(pattern, path)
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    dataset = args.dataset.strip().lower()
    task_type = args.task_type.strip().lower()
    model_name = args.model_name.strip().lower()
    thinking_mode = args.thinking_mode.strip().lower()
    prompt_variant = args.prompt_variant.strip().lower()
    label_set_variant = args.label_set_variant.strip().lower()
    granularity_level = args.granularity_level.strip().lower()
    lang = args.lang.strip().lower()
    batch_size = args.batch_size
    batch_num = args.batch_num
    release_root = args.release_root
    source_images_dir = subset_dir(release_root, task_type)
    reasoning_state = "nonreasoning" if thinking_mode == "off" else "reasoning"

    if task_type == "kie":
        kie_eval_details = run_kie_eval(model_name, dataset, lang, prompt_variant, reasoning_state, release_root)
        print(kie_eval_details)
        os.makedirs(f"evaluations/{dataset}/metrics/kie", exist_ok=True)
        with open(f"evaluations/{dataset}/metrics/kie/{lang}_{model_name}_{prompt_variant}_{reasoning_state}_kie_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(kie_eval_details, f, ensure_ascii=False, indent=2)
        return

    # label_set_variant = labels predicted, granularity_level = labels scored.
    # Valid combos: (full,high), (full,low), (reduced,low); (reduced,high) is invalid.

    # Inference writes predictions under the first segment of --dataset; eval outputs use the full value.
    pred_dataset_root = dataset.split("/")[0]
    prediction_path_pattern = f"predictions/{pred_dataset_root}/dla/{lang}/{model_name}/{prompt_variant}/*{label_set_variant}_labelset_{reasoning_state}_predictions.json"
    
    raw_paths = sorted(glob.glob(prediction_path_pattern))
    parsed = []
    for p in raw_paths:
        start = extract_start_idx(p, task_type)
        if start is not None:
            parsed.append((start, p))
    
    parsed.sort(key=lambda x: x[0])
    prediction_paths = [p for _, p in parsed]

    print(f"No. of batch files: {len(prediction_paths)}")
    
    # load predicted annotations
    merged_predictions = dict()
    for prediction_path in prediction_paths:
        try:
            with open(prediction_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
                for form in predictions:
                    form_id = form.get("form_id")
                    preds = form.get("preds", [])

                    if form_id is None:
                        continue

                    if form_id not in merged_predictions:
                        merged_predictions[form_id] = {"form_id": form_id, "preds": preds}
                    else:
                        merged_predictions[form_id]["preds"].extend(preds)
        except FileNotFoundError:
            print(f"Prediction file not found: {prediction_path}")
    
    combined_predictions = [merged_predictions[k] for k in sorted(merged_predictions.keys())]

    # load target annotations from the export (Label-Studio shape, via release_loader)
    targets = load_dla_targets(os.path.join(source_images_dir, "all_annotations.json"))

    target_annotations = load_sorted_targets(dataset, lang, model_name, targets)
    gt_index = build_gt_index(target_annotations)
    
    predictions = load_sorted_predictions(
        dataset, lang, model_name, task_type,
        prompt_variant, label_set_variant, reasoning_state,
        combined_predictions
    )

    gt_dims = {}
    for t in targets:
        for r in t.get("annotations", [{}])[0].get("result", []):
            if r.get("type") == "rectanglelabels":
                gt_dims[t["id"]] = (r["original_width"], r["original_height"])
                break
    predictions = rescale_predictions_to_gt(predictions, gt_dims)

    pred_index = build_pred_index(predictions)

    for cls in pred_index:
        print(f"{cls}: {len(pred_index[cls])}")

    if not args.no_visualize:
        visualize(task_type, thinking_mode, label_set_variant, predictions, target_annotations, targets, dataset, model_name, prompt_variant, source_images_dir)
    
    CANONICAL_FORM_LABELS, NORMALIZED_FORM_LABELS = preprocess_labels(granularity_level)

    iou_thresholds = [0.3, 0.5, 0.75]

    def _compute_metrics(gt_idx, pred_idx):
        out = {}
        for iou_threshold in iou_thresholds:
            mAP, ap_per_class, _precision, _recall, macro_f1, f1_per_class, overall_avg_iou, thresholded_avg_iou = compute_map_at_iou(
                gt_index=gt_idx,
                pred_index=pred_idx,
                classes=NORMALIZED_FORM_LABELS,
                iou_threshold=iou_threshold,
                granularity_level=granularity_level,
                label_set_variant=label_set_variant,
            )
            ap_per_class = {CANONICAL_FORM_LABELS[k]: float(v) for k, v in ap_per_class.items()}
            f1_per_class = {CANONICAL_FORM_LABELS[k]: float(v) for k, v in f1_per_class.items()}
            out[f"IOU@{iou_threshold}"] = {
                "mAP": float(mAP),
                "macro_f1": macro_f1,
                "overall_avg_iou": overall_avg_iou,
                "thresholded_avg_iou": thresholded_avg_iou,
                "AP_per_class": ap_per_class,
                "F1_per_class": f1_per_class,
            }
        return out

    map_results = _compute_metrics(gt_index, pred_index)

    metrics_dir = f"evaluations/{dataset}/metrics/dla/{lang}/{model_name}"
    os.makedirs(metrics_dir, exist_ok=True)

    map_path = f"{metrics_dir}/{task_type}_annotations_{prompt_variant}_{label_set_variant}_labelset_{granularity_level}_granularity_{reasoning_state}_map_v2.json"

    
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_results, f, indent=2, ensure_ascii=False)
    print(f"mAP results for {model_name} saved to: {map_path}")

    ### Difficulty-bucketed mAP (optional) ###
    if args.difficulty_sheet:
        difficulty_map = load_difficulty_map(args.difficulty_sheet)
        all_form_ids = {t["id"] for t in targets}
        missing = sorted(all_form_ids - set(difficulty_map))
        if missing:
            preview = missing[:10]
            more = f" ... (+{len(missing) - len(preview)} more)" if len(missing) > len(preview) else ""
            print(
                f"⚠️ Difficulty sheet missing {len(missing)} of {len(all_form_ids)} forms; "
                f"excluded from buckets: {preview}{more}"
            )

        bucketed = {"all": {**map_results, "n_forms": len(all_form_ids)}}
        for bucket in DIFFICULTY_BUCKETS:
            bucket_ids = {
                fid for fid, b in difficulty_map.items()
                if b == bucket and fid in all_form_ids
            }
            gt_sub, pred_sub = filter_indices_by_forms(gt_index, pred_index, bucket_ids)
            bucketed[bucket] = {**_compute_metrics(gt_sub, pred_sub), "n_forms": len(bucket_ids)}

        bucketed_path = map_path.replace("_map_v2.json", "_map_v2_by_difficulty.json")
        with open(bucketed_path, "w", encoding="utf-8") as f:
            json.dump(bucketed, f, indent=2, ensure_ascii=False)
        print(f"Bucketed mAP for {model_name} saved to: {bucketed_path}")

if __name__ == "__main__":
    main()