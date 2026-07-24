"""Đánh giá độ chính xác detector — Precision / Recall / F1 + sai số đếm.

Đây là bộ chỉ số ĐÚNG để trả lời "mô hình tốt hay không":

  * Ghép tham lam prediction ↔ ground-truth theo **IoU ≥ ngưỡng** → TP/FP/FN →
    **Precision / Recall / F1**.
  * **MAE/RMSE đếm**: |số box dự đoán − số GT| mỗi ảnh — vì mục tiêu cuối là *đếm*.

KHÔNG dùng mAP: LocateAnything sinh box **không kèm confidence** cho từng box nên
đường PR suy biến. P/R/F1 mới trung thực. (YOLO có confidence thật nên vẫn xếp
hạng được, nhưng để so sánh công bằng ta báo chung P/R/F1 @ IoU.)

Thuần Python, dùng ``BoundingBox.iou`` của ``recognition.base`` → unit-test được
không cần GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .base import BoundingBox

__all__ = ["greedy_match", "DetectionMetrics", "evaluate", "verdict", "nms_dedup"]


def nms_dedup(
    boxes: Sequence[BoundingBox],
    scores: Optional[Sequence[float]] = None,
    iou_thr: float = 0.9,
) -> List[int]:
    """NMS/khử trùng lặp (class-agnostic). Trả danh sách chỉ số box GIỮ lại.

    LocateAnything greedy-decode dễ lặp box → nhiều box gần y hệt. Dedup trước khi
    chấm điểm để precision phản ánh đúng. Nếu confidence bằng nhau (model không
    phát điểm tin cậy) thì ưu tiên giữ box lớn hơn (tất định).
    """
    n = len(boxes)
    if n == 0:
        return []
    if scores is not None and len(set(scores)) > 1:
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    else:
        order = sorted(range(n), key=lambda i: boxes[i].area, reverse=True)
    keep: List[int] = []
    removed = [False] * n
    for pos, i in enumerate(order):
        if removed[i]:
            continue
        keep.append(i)
        for j in order[pos + 1:]:
            if not removed[j] and boxes[i].iou(boxes[j]) >= iou_thr:
                removed[j] = True
    return sorted(keep)


def greedy_match(
    pred_boxes: Sequence[BoundingBox],
    gt_boxes: Sequence[BoundingBox],
    iou_thr: float = 0.5,
    scores: Optional[Sequence[float]] = None,
) -> Tuple[int, int, int, List[float]]:
    """Ghép tham lam pred ↔ GT theo IoU. Trả ``(tp, fp, fn, iou_của_TP)``.

    Duyệt prediction theo confidence giảm dần (nếu có ``scores``); mỗi pred bắt
    cặp với GT **chưa dùng** có IoU cao nhất & ≥ ``iou_thr`` → TP, ngược lại FP.
    Mỗi GT khớp tối đa 1 pred (chuẩn COCO cho box thường).
    """
    order = list(range(len(pred_boxes)))
    if scores is not None:
        order.sort(key=lambda i: scores[i], reverse=True)

    used = [False] * len(gt_boxes)
    tp = fp = 0
    tp_iou: List[float] = []
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if used[j]:
                continue
            v = pred_boxes[i].iou(g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            tp += 1
            tp_iou.append(best_iou)
        else:
            fp += 1
    fn = len(gt_boxes) - sum(used)
    return tp, fp, fn, tp_iou


@dataclass
class DetectionMetrics:
    """Chỉ số gộp trên nhiều ảnh cho một lớp."""

    iou_thr: float = 0.5
    label: str = ""
    n_images: int = 0
    n_pred: int = 0
    n_gt: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tp_iou_sum: float = 0.0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    within1: int = 0
    bias_sum: int = 0

    def add_image(self, pred_boxes, gt_boxes, scores=None) -> None:
        tp, fp, fn, ious = greedy_match(pred_boxes, gt_boxes, self.iou_thr, scores)
        self.n_images += 1
        self.n_pred += len(pred_boxes)
        self.n_gt += len(gt_boxes)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tp_iou_sum += sum(ious)
        err = len(pred_boxes) - len(gt_boxes)
        self.abs_err_sum += abs(err)
        self.sq_err_sum += err * err
        self.within1 += int(abs(err) <= 1)
        self.bias_sum += err

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_tp_iou(self) -> float:
        return self.tp_iou_sum / self.tp if self.tp else 0.0

    @property
    def count_mae(self) -> float:
        return self.abs_err_sum / self.n_images if self.n_images else 0.0

    @property
    def count_rmse(self) -> float:
        return math.sqrt(self.sq_err_sum / self.n_images) if self.n_images else 0.0

    @property
    def count_within1_rate(self) -> float:
        return self.within1 / self.n_images if self.n_images else 0.0

    @property
    def count_bias(self) -> float:
        return self.bias_sum / self.n_images if self.n_images else 0.0

    def as_row(self) -> dict:
        return {
            "lớp": self.label,
            "ảnh": self.n_images,
            "P": round(self.precision, 3),
            "R": round(self.recall, 3),
            "F1": round(self.f1, 3),
            "GT": self.n_gt,
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "MAE": round(self.count_mae, 2),
            "kết luận": verdict(self.f1),
        }


def evaluate(
    preds_per_image: Sequence[Sequence[BoundingBox]],
    gts_per_image: Sequence[Sequence[BoundingBox]],
    iou_thr: float = 0.5,
    label: str = "",
) -> DetectionMetrics:
    """Gộp đánh giá trên danh sách ảnh (pred/gt cùng độ dài)."""
    assert len(preds_per_image) == len(gts_per_image), "số ảnh pred/gt phải bằng nhau"
    m = DetectionMetrics(iou_thr=iou_thr, label=label)
    for preds, gts in zip(preds_per_image, gts_per_image):
        m.add_image(preds, gts)
    return m


def verdict(f1: float) -> str:
    """Diễn giải nhanh chất lượng theo F1@0.5 (tham khảo)."""
    if f1 >= 0.80:
        return "🟢 Rất tốt (dùng thẳng)"
    if f1 >= 0.65:
        return "🟢 Tốt (dùng được)"
    if f1 >= 0.45:
        return "🟡 Khá (chỉnh prompt/NMS hoặc cân nhắc fine-tune)"
    if f1 >= 0.25:
        return "🟠 Yếu (nên fine-tune domain của bạn)"
    return "🔴 Kém (prompt/dataset chưa khớp hoặc cần fine-tune)"
