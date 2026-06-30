import json
import os
import re
import base64
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image


# model_family -> {model_name -> OpenRouter model path} for sequential inference.
OPENROUTER_MODEL_PATH_MAP = {
    "openai": {
        "gpt_5.1": "gpt-5.1",
        "gpt_5_mini": "openai/gpt-5-mini",
        "gpt_5.2": "gpt-5.2",
        "gpt_5.4": "openai/gpt-5.4",
        "gpt_4": "gpt-4.1"
    },
    "qwen": {
        "qwen_2.5_vl": "qwen/qwen2.5-vl-72b-instruct",
        "qwen_3_vl": "qwen/qwen3-vl-32b-instruct",
        "qwen_3.5": "qwen/qwen3.5-plus-02-15",
        "qwen_3.5_27b": "qwen/qwen3.5-27b",
        "qwen_3.6": "qwen/qwen3.6-plus"
    },
    "gemini": {
        "gemini_2.5_flash": "google/gemini-2.5-flash",
        "gemini_2.5_pro": "google/gemini-2.5-pro",
        "gemini_3": "google/gemini-3.1-pro-preview",
        "gemini_3.5_flash": "google/gemini-3.5-flash"
    },
    "gemma": {
        "gemma_4": "google/gemma-3-4b-it",
        "gemma_12": "google/gemma-3-12b-it",
        "gemma_27": "google/gemma-3-27b-it",
        "gemma4_31b": "google/gemma-4-31b-it",
        "gemma4_31b_free": "google/gemma-4-31b-it:free"
    },
    "claude": {
        "claude_opus_4.5": "anthropic/claude-opus-4-5",
        "claude_opus_4.6": "anthropic/claude-opus-4-6"
    },
    "kimi": {
        "kimi_k2.5": "moonshotai/kimi-k2.5"
    },
    "mistral": {
        "pixtral_large": "mistralai/pixtral-large-2411",
        "mistral_small_3.2": "mistralai/mistral-small-3.2-24b-instruct",
        "mistral_small_4": "mistralai/mistral-small-2603"
    },
    "glm": {
        "glm_4.6v": "z-ai/glm-4.6v"
    }
}

NATIVE_MODEL_PATH_MAP = {
    "openai": {
        "gpt_5.1": "gpt-5.1",
        "gpt_5.2": "gpt-5.2",
        "gpt_4": "gpt-4.1"
    },
    "qwen": {
        "qwen_2.5_vl": "qwen/qwen2.5-vl-72b-instruct",
        "qwen_3_vl": "qwen3-vl-plus"
    },
    "gemini": {
        "gemini_2.5_pro": "gemini-2.5-pro",
        "gemini_3": "gemini-3.1-pro-preview"
    },
    "claude": {
        "claude_opus_4.5": "claude-opus-4-5",
        "claude_opus_4.6": "claude-opus-4-6"
    }
}

LONG_FORM_LABEL_CATEGORIES = [
    "Header",
    "Title of Form",
    "Section Title",
    "Footer",
    "Page Num",
    "Form Key",
    "Form Value",
    "Inline Key",
    "Inline Value",
    "Checkbox",
    "Tick Mark",
    "Signature Key",
    "Signature Val",
    "Photo Field",
    "Figure/Diagram/Logo",
    "Table",
    "Table Caption",
    "Table Section Title",
    "Table Index",
    "Table Col PKey",
    "Table Col Value",
    "Table Row PKey",
    "Table Row Value",
    "Text Block",
    "Gibberish, Mark for removal",
    "Others",
]

SHORT_FORM_LABEL_CATEGORIES = [
    "Headings",
    "Table",
    "Image",
    "Fields",
    "Others"
]

SHORT_TO_LONG_LABEL_MAP = {
    "headings": (
        "header", 
        "title of form", 
        "footer", 
        "section title"
    ),
    "table": (
        "table",
    ),
    "image": ("figure/diagram/logo",),
    "fields": (
        "form key", 
        "form value", 
        "inline key", 
        "inline value", 
        "signature key", 
        "signature val", 
        "tick mark", 
        "checkbox", 
        "photo field"
    ),
    "others": (
        "text block", 
        "page num", 
        "gibberish, mark for removal", 
        "others"
    )
}


def encode_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        return encoded_string
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

def load_form_pages(subset_dir, form):
    """Return (form_id, path, b64, height, width) for each readable page of a form, in order."""
    form_id = form["id"]
    form_page_images = []
    for rel in form.get("local_images", []):
        path = os.path.join(subset_dir, *rel.split("/"))
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
        with Image.open(path) as img:
            width, height = img.size
        img_b64 = encode_image(path)
        if img_b64 is None:
            print(f"⚠️ Could not read image file: {path}")
            continue
        form_page_images.append((form_id, path, img_b64, height, width))
    return form_page_images


def normalize_label(s):
    return s.strip().lower()


def preprocess_labels(granularity_level):
    if granularity_level == "low":
        CANONICAL_FORM_LABELS = {normalize_label(c): c for c in SHORT_FORM_LABEL_CATEGORIES}
        NORMALIZED_FORM_LABELS = list(CANONICAL_FORM_LABELS.keys())
    elif granularity_level == "high":
        CANONICAL_FORM_LABELS = {normalize_label(c): c for c in LONG_FORM_LABEL_CATEGORIES}
        NORMALIZED_FORM_LABELS = list(CANONICAL_FORM_LABELS.keys())

    return CANONICAL_FORM_LABELS, NORMALIZED_FORM_LABELS


def parse_json(text, return_meta=False):
    """Parse JSON from `text`; with `return_meta=True` also returns `{"used_salvage": bool}`."""
    def _wrap(parsed, used_salvage):
        return (parsed, {"used_salvage": used_salvage}) if return_meta else parsed

    if not text:
        print("⚠️ Empty response text")
        return _wrap(None, False)

    pattern = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if pattern:
        parsed_text = pattern.group(1).strip()
    else:
        parsed_text = text.strip()

    if not (parsed_text.startswith("{") or parsed_text.startswith("[")):
        start = parsed_text.find("{")
        if start == -1:
            start = parsed_text.find("[")
        if start != -1:
            parsed_text = parsed_text[start:]

    try:
        parsed_json = json.loads(parsed_text)
        return _wrap(parsed_json, False)
    except json.JSONDecodeError as e:
        # Salvage path: recover concatenated top-level objects when the leading `[{` is missing.
        try:
            decoder = json.JSONDecoder()
            objs = []
            text = parsed_text.lstrip("[").rstrip("]").strip()
            i = 0
            while i < len(text):
                while i < len(text) and text[i] in ", \n\r\t":
                    i += 1
                if i >= len(text):
                    break
                obj, end = decoder.raw_decode(text, i)
                objs.append(obj)
                i = end
            if objs:
                return _wrap(objs, True)
        except Exception:
            pass
        print(f"⚠️ Failed to Parse JSON Output {e}")
        return _wrap([], False)


def denormalize_bbox(normalized_bbox, original_width, original_height):
    x = normalized_bbox['x'] / 100 * original_width
    y = normalized_bbox['y'] / 100 * original_height
    w = normalized_bbox['width'] / 100 * original_width
    h = normalized_bbox['height'] / 100 * original_height
    return dict(x=x, y=y, width=w, height=h)


def format_annotations(targets):
    formatted_targets = []

    for form_idx, form_annotations in enumerate(targets):
        compact_target = dict()
        compact_target["form_id"] = form_annotations["id"]
        page_count = len(form_annotations["data"]["pages"])
        compact_target['total_pages'] = page_count
        compact_target["bbox_details"] = [[] for _ in range(page_count)]
        for _, annotation_details in enumerate(form_annotations["annotations"][0]["result"]):
            # avoid entity linking annotations
            if annotation_details["type"] != "rectanglelabels":
                continue
            bbox_details = dict()
            bbox_details["bb_box_id"] = annotation_details["id"]
            bbox_details["original_width"] = annotation_details["original_width"]
            bbox_details["original_height"] = annotation_details["original_height"]
            
            normalized_bbox={ 
                "x": annotation_details["value"]["x"],
                "y": annotation_details["value"]["y"],
                "width": annotation_details["value"]["width"],
                "height": annotation_details["value"]["height"],
            }
            bbox_details["denormalized_coordinates"] = denormalize_bbox(normalized_bbox, annotation_details["original_width"], annotation_details["original_height"]) 
            bbox_details["label"] = annotation_details["value"]["rectanglelabels"][0]     
            
            page_idx = annotation_details.get("item_index", 0)
            if 0 <= page_idx < page_count:
                compact_target["bbox_details"][page_idx].append(bbox_details)
            else:
                print(f"Warning: invalid page_idx {page_idx} for form {form_annotations['id']}")
        formatted_targets.append(compact_target)

    return formatted_targets


def build_gt_index(target_annotations):
    gt_index = defaultdict(lambda: defaultdict(list))

    for annotation in target_annotations:
        form_id = annotation["form_id"]

        for page_idx, page_boxes in enumerate(annotation.get("sorted_bbox_details", [])):
            for box in page_boxes:
                label = normalize_label(box.get("label", ""))
                coords = box["denormalized_coordinates"]
                bbox_xywh = [
                    float(coords["x"]),
                    float(coords["y"]),
                    float(coords["width"]),
                    float(coords["height"]),
                ]
                gt_index[(form_id, page_idx)][label].append(bbox_xywh)

    return gt_index


def build_pred_index(predictions):
    pred_index = defaultdict(list)

    for form in predictions:
        form_id = form["form_id"]
        for page_idx, page_preds in enumerate(form.get("sorted_preds", [])):
            for p in page_preds:
                cls = normalize_label(p.get("class", ""))
                conf_score = float(p.get("confidence", 0.0))
                bbox = [float(x) for x in p["bbox"]]  # [x,y,w,h]
                pred_index[cls].append((form_id, page_idx, conf_score, bbox))

    for cls in pred_index:
        pred_index[cls].sort(key=lambda x: x[2], reverse=True)

    return pred_index


def filter_indices_by_forms(gt_index, pred_index, form_ids):
    """Restrict (gt_index, pred_index) to `form_ids`, keeping every pred class key (even if emptied)."""
    form_ids = set(form_ids)
    gt_sub = defaultdict(lambda: defaultdict(list))
    for (form_id, page_idx), label_to_boxes in gt_index.items():
        if form_id in form_ids:
            gt_sub[(form_id, page_idx)] = label_to_boxes

    pred_sub = defaultdict(list)
    for cls, entries in pred_index.items():
        pred_sub[cls] = [e for e in entries if e[0] in form_ids]
    return gt_sub, pred_sub


def rescale_predictions_to_gt(sorted_predictions, gt_dims, fallback_clamp_max=10.0):
    """Rescale each form's pred bboxes into GT pixel space from its own max (x+w, y+h) extent."""
    for form_preds in sorted_predictions:
        form_id = form_preds.get("form_id")
        if form_id not in gt_dims:
            continue
        gw, gh = gt_dims[form_id]

        max_x = 0.0
        max_y = 0.0
        for page in form_preds.get("sorted_preds", []):
            for box in page:
                bx, by, bw, bh = box["bbox"]
                if bx + bw > max_x:
                    max_x = bx + bw
                if by + bh > max_y:
                    max_y = by + bh

        if max_x <= 0 or max_y <= 0:
            continue

        sx = gw / max_x
        sy = gh / max_y

        if sx > fallback_clamp_max or sy > fallback_clamp_max:
            continue

        for page in form_preds.get("sorted_preds", []):
            for box in page:
                bx, by, bw, bh = box["bbox"]
                box["bbox"] = [bx * sx, by * sy, bw * sx, bh * sy]

    return sorted_predictions


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def greedy_matching(gt_index, pred_index, cls, iou_threshold, granularity_level, label_set_variant):
    # Direct access, except full (26) predictions against reduced (5) targets, which collapse sub-labels.
    if (granularity_level == "high" and label_set_variant == "full") or (granularity_level == "low" and label_set_variant == "reduced"):
        preds_cls = pred_index.get(cls, [])
    elif (granularity_level == "low" and label_set_variant == "full"):
        preds_cls = []
        for subclass in SHORT_TO_LONG_LABEL_MAP[cls]:
            preds_cls.extend(pred_index.get(subclass, []))
        preds_cls.sort(key=lambda x: x[2], reverse=True)  # re-sort by confidence after merging sub-labels

    tp = [0] * len(preds_cls)
    fp = [0] * len(preds_cls)

    overall_iou_sum, overall_bbox_cnt = 0.0, 0
    thresholded_iou_sum, thresholded_bbox_cnt = 0.0, 0

    n_gt = 0
    matched = dict()

    for (form_id, page_idx), classwise_annotations in gt_index.items():
        gts = []
        if (granularity_level == "high" and label_set_variant == "full"):
            gts = classwise_annotations.get(cls, [])
        elif (granularity_level == "low" and label_set_variant == "full") or (granularity_level == "low" and label_set_variant == "reduced"):
            for class_name, class_annotations in classwise_annotations.items():
                if class_name in SHORT_TO_LONG_LABEL_MAP[cls]:
                    gts.extend(class_annotations)
        
        if gts:
            matched[(form_id, page_idx)] = [False] * len(gts)
            n_gt += len(gts)

    for idx, (form_id, page_idx, conf_score, pred_bbox) in enumerate(preds_cls):
        key = (form_id, page_idx)
        gt_annotations_per_page = dict()
        if (granularity_level == "high" and label_set_variant == "full"):
            gt_annotations_per_page = gt_index.get(key, {})
        elif (granularity_level == "low" and label_set_variant == "full") or (granularity_level == "low" and label_set_variant == "reduced"):
            gt_annotations_per_page[cls] = []
            for gt_class, gt_class_annotations in gt_index.get(key, {}).items():
                if gt_class in SHORT_TO_LONG_LABEL_MAP[cls]:
                    gt_annotations_per_page[cls].extend(gt_class_annotations)
    
        gt_bboxes = gt_annotations_per_page.get(cls, [])

        if len(gt_bboxes) == 0:
            fp[idx] = 1
            continue

        used_flags = matched.get(key)

        if used_flags is None:
            used_flags = [False] * len(gt_bboxes)
            matched[key] = used_flags

        best_iou = 0.0
        best_jdx = -1
        
        for jdx, gt_bbox in enumerate(gt_bboxes):
            if used_flags[jdx]:
                continue
            curr_iou = iou_xywh(pred_bbox, gt_bbox)
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_jdx = jdx
        
        overall_bbox_cnt +=1
        overall_iou_sum += best_iou
        if best_iou >= iou_threshold and best_jdx >= 0:
            thresholded_bbox_cnt += 1
            thresholded_iou_sum += best_iou
            tp[idx] = 1
            used_flags[best_jdx] = True
        else:
            fp[idx] = 1
            
    overall_class_avg_iou = overall_iou_sum / overall_bbox_cnt if overall_bbox_cnt > 0 else 0.0
    thresholded_class_avg_iou = thresholded_iou_sum / thresholded_bbox_cnt if thresholded_bbox_cnt > 0 else 0.0
    return tp, fp, n_gt, overall_class_avg_iou, thresholded_class_avg_iou, thresholded_bbox_cnt


def calculate_average_precision(tp, fp, n_gt, n_recall_points=101):
    if n_gt == 0:
        return np.array([]), np.array([]), None

    if len(tp) == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32), 0.0

    tp = np.asarray(tp, dtype=np.float32)
    fp = np.asarray(fp, dtype=np.float32)

    cumul_tp = np.cumsum(tp)
    cumul_fp = np.cumsum(fp)

    # 1e-12 is used to avoid division by zero
    precision = cumul_tp / np.maximum(cumul_tp + cumul_fp, 1e-12)
    recall = cumul_tp / float(n_gt)

    # Interpolated precision envelope: max precision at recall >= r
    precision_env = np.maximum.accumulate(precision[::-1])[::-1]

    # 101-point sampling, following COCO and Form-NLU
    recall_samples = np.linspace(0.0, 1.0, n_recall_points, dtype=np.float32)

    precision_samples = np.zeros_like(recall_samples)
    for idx, r in enumerate(recall_samples):
        indices = np.where(recall >= r)[0]
        precision_samples[idx] = precision_env[indices[0]] if len(indices) > 0 else 0.0

    ap = float(np.mean(precision_samples))
    return precision, recall, ap


def compute_map_at_iou(gt_index, pred_index, classes, iou_threshold, granularity_level, label_set_variant):
    ap_per_class = dict()
    ap_values = []

    overall_avg_iou_details = dict()
    overall_avg_iou_values = []

    f1_scores = []
    f1_per_class = dict()

    thresholded_avg_iou_details = dict()
    thresholded_avg_iou_values = []

    for idx, cls in enumerate(classes):
        tp, fp, n_gt, overall_class_avg_iou, thresholded_class_avg_iou, thresholded_bbox_cnt = greedy_matching(gt_index, pred_index, cls, iou_threshold, granularity_level, label_set_variant)
        precision, recall, ap = calculate_average_precision(tp, fp, n_gt)
        if len(precision) > 0 and len(recall) > 0:
            p = precision[-1]
            r = recall[-1]
            f1 = (2 * p * r) / (p + r) if (p + r) > 0.0 else 0.0
        else:
            f1 = 0.0
        f1_scores.append(float(f1))
        f1_per_class[cls] = float(f1)

        if n_gt > 0:
            overall_avg_iou_details[cls] = overall_class_avg_iou
            overall_avg_iou_values.append(overall_class_avg_iou)

        if thresholded_bbox_cnt > 0:
            thresholded_avg_iou_details[cls] = thresholded_class_avg_iou
            thresholded_avg_iou_values.append(thresholded_class_avg_iou)
        
        if ap is None:
            continue

        ap_per_class[cls] = float(ap)
        ap_values.append(float(ap))

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mAP = float(sum(ap_values) / len(ap_values)) if ap_values else 0.0
    overall_avg_iou = sum(overall_avg_iou_values) / len(overall_avg_iou_values) if len(overall_avg_iou_values) > 0 else 0.0
    thresholded_avg_iou = sum(thresholded_avg_iou_values) / len(thresholded_avg_iou_values) if len(thresholded_avg_iou_values) > 0 else 0.0
    return mAP, ap_per_class, precision, recall, macro_f1, f1_per_class, overall_avg_iou, thresholded_avg_iou


def get_versioned_path(base_path: str | Path) -> Path:
    base_path = Path(base_path)
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    i = 1
    while True:
        candidate = parent / f"{stem}{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def remove_wrapper(parsed_output):
    output = parsed_output
    
    POSSIBLE_WRAPPER_KEYS = (
        "field",
        "fields", 
        "layout",
        "element",
        "elements",
        "item", 
        "items", 
        "output",
        "outputs",
        "pred", 
        "preds", 
        "result",
        "results"
    )

    if isinstance(output, dict):
        for k in POSSIBLE_WRAPPER_KEYS:
            v = output.get(k)
            if isinstance(v, list):
                output = v
                break
        
    if not isinstance(output, list) and isinstance(output, dict):
        if len(output) == 1:
            only_val = next(iter(output.values()))
            if isinstance(only_val, list):
                output = only_val

    return output
