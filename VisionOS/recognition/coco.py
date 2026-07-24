"""Nạp **COCO val2017** để đánh giá độ chính xác mô hình trên dữ liệu thật có nhãn.

Lớp COCO khớp trực tiếp 3 bài toán đếm của sản phẩm:
  * ``person`` → đếm người,   * ``car`` → đếm xe,   * ``bottle`` → đếm sản phẩm.

Phần *parse ground-truth* là hàm thuần (unit-test được); phần *tải* cần mạng
(chạy trên Kaggle). Nếu đã "Add Data" COCO trên Kaggle, trỏ ``ann_file`` /
``images_dir`` để khỏi tải lại.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "EVAL_CLASSES",
    "coco_bbox_to_xyxy",
    "parse_coco_gt",
    "select_images_with_category",
    "ensure_coco_val_annotations",
    "download_image",
    "load_eval_samples",
]

Box = Tuple[float, float, float, float]

# 1 "lớp đánh giá" = prompt gửi cho model + tập danh mục COCO tính là ground-truth.
EVAL_CLASSES: Dict[str, dict] = {
    "person": {"prompt": "person", "coco_cats": ["person"], "task": "đếm người"},
    "car":    {"prompt": "car",    "coco_cats": ["car"],    "task": "đếm xe"},
    "bottle": {"prompt": "bottle", "coco_cats": ["bottle"], "task": "đếm sản phẩm"},
    "vehicle": {"prompt": "vehicle", "coco_cats": ["car", "truck", "bus", "motorcycle"],
                "task": "đếm phương tiện (gộp)"},
}

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def coco_bbox_to_xyxy(bbox: Sequence[float]) -> Box:
    """COCO bbox [x, y, w, h] → (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def _cat_ids_for(coco: dict, names: Sequence[str]) -> set:
    name2id = {c["name"]: c["id"] for c in coco.get("categories", [])}
    return {name2id[n] for n in names if n in name2id}


def parse_coco_gt(
    coco: dict, coco_cat_names: Sequence[str], include_crowd: bool = False
) -> Dict[int, List[Box]]:
    """Gom GT box theo image_id cho các danh mục cần. Bỏ ``iscrowd=1`` (chuẩn COCO)."""
    cat_ids = _cat_ids_for(coco, coco_cat_names)
    gt: Dict[int, List[Box]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("category_id") not in cat_ids:
            continue
        if not include_crowd and ann.get("iscrowd", 0) == 1:
            continue
        gt.setdefault(ann["image_id"], []).append(coco_bbox_to_xyxy(ann["bbox"]))
    return gt


def select_images_with_category(
    coco: dict, coco_cat_names: Sequence[str], limit: int = 100, include_crowd: bool = False
) -> List[dict]:
    """Chọn tối đa ``limit`` ảnh CÓ chứa danh mục (tất định theo image_id)."""
    gt = parse_coco_gt(coco, coco_cat_names, include_crowd)
    imgs = [im for im in coco.get("images", []) if im["id"] in gt]
    imgs.sort(key=lambda im: im["id"])
    return imgs[:limit]


# --------------------------------------------------------------------------- #
# Phần cần mạng (Kaggle) — không unit-test
# --------------------------------------------------------------------------- #
def ensure_coco_val_annotations(
    cache_dir: str = "/kaggle/working/coco", ann_file: Optional[str] = None
) -> str:
    """Đảm bảo có ``instances_val2017.json``; tải + giải nén nếu thiếu."""
    if ann_file and os.path.exists(ann_file):
        return ann_file
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, "instances_val2017.json")
    if os.path.exists(target):
        return target

    import urllib.request
    import zipfile

    zip_path = os.path.join(cache_dir, "annotations_trainval2017.zip")
    if not os.path.exists(zip_path):
        print(f"📥 Tải annotations COCO (~241MB): {COCO_ANN_URL}")
        urllib.request.urlretrieve(COCO_ANN_URL, zip_path)
    print("📦 Giải nén instances_val2017.json ...")
    with zipfile.ZipFile(zip_path) as z:
        with z.open("annotations/instances_val2017.json") as s, open(target, "wb") as d:
            d.write(s.read())
    try:
        os.remove(zip_path)
    except OSError:
        pass
    return target


def download_image(img: dict, images_dir: str) -> Optional[str]:
    """Tải 1 ảnh COCO theo ``coco_url`` (bỏ qua nếu đã có)."""
    os.makedirs(images_dir, exist_ok=True)
    path = os.path.join(images_dir, img["file_name"])
    if os.path.exists(path):
        return path
    import urllib.request

    url = img.get("coco_url") or f"http://images.cocodataset.org/val2017/{img['file_name']}"
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠️ tải lỗi {img['file_name']}: {e}")
        return None


def load_eval_samples(
    eval_class: str,
    limit: int = 50,
    cache_dir: str = "/kaggle/working/coco",
    ann_file: Optional[str] = None,
    images_dir: Optional[str] = None,
) -> List[Tuple[str, List[Box]]]:
    """Trả ``[(image_path, gt_boxes_xyxy), ...]`` cho một lớp đánh giá."""
    if eval_class not in EVAL_CLASSES:
        raise ValueError(f"eval_class {eval_class!r} không hợp lệ. Chọn: {list(EVAL_CLASSES)}")
    cats = EVAL_CLASSES[eval_class]["coco_cats"]

    ann_path = ensure_coco_val_annotations(cache_dir, ann_file)
    with open(ann_path) as f:
        coco = json.load(f)

    gt_map = parse_coco_gt(coco, cats)
    imgs = select_images_with_category(coco, cats, limit)
    images_dir = images_dir or os.path.join(cache_dir, "val2017")

    samples: List[Tuple[str, List[Box]]] = []
    for im in imgs:
        path = download_image(im, images_dir)
        if path:
            samples.append((path, gt_map.get(im["id"], [])))
    return samples
