import json
import os
import shutil


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def subset_dir(release_root, task):
    """Directory of the ``task`` ('dla'|'kie') subset inside a downloaded export root."""
    return os.path.join(release_root, f"bafco_{task}")


def load_forms(release_root, task):
    """Every form record for ``task`` from ``<release_root>/bafco_<task>/all_annotations.json``."""
    return _load(os.path.join(subset_dir(release_root, task), "all_annotations.json"))


def load_dla_targets(export_path):
    targets = []
    for f in _load(export_path):
        result = []
        for pg in f.get("pages", []):
            ow, oh = pg.get("original_width"), pg.get("original_height")
            for b in pg.get("boxes", []):
                result.append({
                    "type": "rectanglelabels",
                    "id": b.get("id"),
                    "original_width": ow,
                    "original_height": oh,
                    "item_index": pg.get("page", 0),
                    "image_rotation": 0,
                    "value": {
                        "x": b.get("x"), "y": b.get("y"),
                        "width": b.get("width"), "height": b.get("height"),
                        "rotation": b.get("rotation", 0),
                        "rectanglelabels": [b.get("label")],
                    },
                })
        for r in f.get("relations", []):
            result.append({"type": "relation", "from_id": r.get("from_id"),
                           "to_id": r.get("to_id"), "direction": r.get("direction")})
        targets.append({
            "id": f["id"],
            "inner_id": f.get("inner_id", f["id"]),
            "data": {"pages": list(f.get("local_images", []))},
            "annotations": [{"result": result}],
        })
    return targets


def load_kie_forms(export_path):
    forms = _load(export_path)
    for f in forms:
        if "id" not in f or "annotations" not in f:
            raise ValueError(f"KIE form missing id/annotations: {f.get('id')}")
        for pg in f["annotations"]:
            if "page" not in pg or "annotations" not in pg:
                raise ValueError(f"KIE form {f['id']} page block missing page/annotations")
    return forms


def resolve_release_image(export_root, form, page_index):
    rels = form.get("local_images") or []
    if not 0 <= page_index < len(rels):
        raise IndexError(f"page {page_index} out of range for form {form.get('id')}")
    return os.path.join(export_root, *rels[page_index].split("/"))


def stage_kie_flat_images(export_path, dest_dir):
    export_root = os.path.dirname(os.path.abspath(export_path))
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for f in _load(export_path):
        for page_index, rel in enumerate(f.get("local_images", [])):
            src = os.path.join(export_root, *rel.split("/"))
            ext = os.path.splitext(rel)[1] or ".jpg"
            shutil.copy2(src, os.path.join(dest_dir, f"{f['id']}_{page_index}{ext}"))
            n += 1
    return n
