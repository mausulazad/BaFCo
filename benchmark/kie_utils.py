from typing import Dict, List, Tuple
import re
import string

from rapidfuzz.distance import Levenshtein
import sacrebleu

def normalize_text(text: str) -> str:
    if len(text) == 0:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

# Exact Match (EM)
def calculate_em(pred: str, gt: str) -> float:
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    return 1.0 if pred_norm == gt_norm else 0.0

# Normalized Edit Similarity (NES)
def calculate_nes(pred: str, gt: str) -> float:
    if not pred and not gt:
        return 1.0

    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return 1.0

    dist = Levenshtein.distance(pred, gt)
    return 1 - float(dist / max_len)

# Character Detection Metric (CDM)
def calculate_cdm(pred: str, gt: str) -> Tuple:
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)

    len_pred = len(pred_norm)
    len_gt = len(gt_norm)

    if len_pred == 0 and len_gt == 0:
        return (1.0, 1.0, 1.0)

    if len_pred == 0 or len_gt == 0:
        return (0.0, 0.0, 0.0)

    dist = Levenshtein.distance(pred_norm, gt_norm)
    correct = max(len_gt - dist, 0)

    precision = correct / len_pred if len_pred > 0 else 0.0
    recall = correct / len_gt if len_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (float(precision), float(recall), float(f1))


def calculate_token_f1(pred: str, gt: str) -> Tuple[float, float, float]:
    def tokenize(text: str) -> List[str]:
        text = normalize_text(text).lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    pred_tokens = tokenize(pred)
    gt_tokens = tokenize(gt)

    if not pred_tokens and not gt_tokens:
        return (1.0, 1.0, 1.0)
    if not pred_tokens or not gt_tokens:
        return (0.0, 0.0, 0.0)

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gt_counts = {}
    for t in gt_tokens:
        gt_counts[t] = gt_counts.get(t, 0) + 1

    overlap = sum(min(pred_counts.get(t, 0), gt_counts[t]) for t in gt_counts)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return (float(precision), float(recall), float(f1))


def calculate_bleu(pred: str, gt: str) -> float:
    len_pred, len_gt = len(pred), len(gt)
    if len_pred == 0 and len_gt == 0:
        return 1.0

    if len_pred == 0 or len_gt == 0:
        return 0.0
    
    bleu = sacrebleu.sentence_bleu(
        pred, 
        [gt],
        tokenize="none", 
        smooth_method="exp"
    )
    return float(bleu.score) / 100.0

def get_kie_eval(predictions: List[str], targets: List[str]) -> Dict:
    """Compute all KIE metrics over paired prediction/target lists. Returns averaged scores as a dict."""
    em_scores, nes_scores = [], []
    cdm_precisions, cdm_recalls, cdm_f1s = [], [], []
    bleu_scores = []
    token_precisions, token_recalls, token_f1s = [], [], []

    for pred, gt in zip(predictions, targets):
        em_scores.append(calculate_em(pred, gt))
        nes_scores.append(calculate_nes(pred, gt))
        cp, cr, cf1 = calculate_cdm(pred, gt)
        cdm_precisions.append(cp)
        cdm_recalls.append(cr)
        cdm_f1s.append(cf1)
        bleu_scores.append(calculate_bleu(pred, gt))
        tp, tr, tf1 = calculate_token_f1(pred, gt)
        token_precisions.append(tp)
        token_recalls.append(tr)
        token_f1s.append(tf1)

    total = len(em_scores)
    if total == 0:
        raise ValueError("No predictions to evaluate — check lang filter and prediction file contents.")
    return {
        "em_score":        float(sum(em_scores)) / total,
        "nes_score":       float(sum(nes_scores)) / total,
        "precision":       float(sum(cdm_precisions)) / total,
        "recall":          float(sum(cdm_recalls)) / total,
        "cdm_score":       float(sum(cdm_f1s)) / total,
        "bleu_score":      float(sum(bleu_scores)) / total,
        "token_precision": float(sum(token_precisions)) / total,
        "token_recall":    float(sum(token_recalls)) / total,
        "token_f1":        float(sum(token_f1s)) / total,
        "total":           total,
    }