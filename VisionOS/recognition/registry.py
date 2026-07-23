"""Sổ đăng ký mô hình — ánh xạ nghiệp vụ ↔ mô hình nhận diện.

Gom về một chỗ mọi ánh xạ "nghiệp vụ nào chạy mô hình nào" mà frontend đang mô tả
rải rác (``STANDARD_MODEL_NAMES``, các tag ``AI_USECASES``). Nhờ vậy backend có
một nguồn sự thật duy nhất để hiển thị tên mô hình và khởi tạo detector đúng.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .base import MonitoringMode

__all__ = [
    "ModelSpec",
    "STANDARD_MODEL_NAMES",
    "TASK_LABELS",
    "MODEL_SPECS",
    "resolve_model",
    "spec_for",
]

# Tên mô hình theo nghiệp vụ ở chế độ STANDARD (khớp PipelineBuilder.tsx).
STANDARD_MODEL_NAMES: Dict[str, str] = {
    "security": "YOLO-NAS-S · Phát hiện người",
    "counting": "YOLO-NAS-S · Đếm đối tượng",
    "ppe": "YOLO-NAS + PPE Classifier",
    "fire": "FireNet v2 · Khói & Lửa",
    "traffic": "YOLO-NAS + ALPR",
    "behavior": "YOLO-NAS + Action Recognition",
    "retail_analytics": "YOLO-NAS + Heatmap Analytics",
}

# Nhãn nghiệp vụ hiển thị (khớp TASK_LABELS ở frontend).
TASK_LABELS: Dict[str, str] = {
    "security": "Giám sát An ninh",
    "counting": "Đếm lưu lượng",
    "defect_surface": "Lỗi bề mặt",
    "defect_assembly": "Lỗi lắp ráp & Đóng gói",
    "defect_label": "Kiểm tra tem nhãn",
    "defect_foreign": "Phát hiện dị vật",
    "label_inspection": "Kiểm tra tem nhãn / Hạn dùng",
    "assembly_inspection": "Lỗi lắp ráp",
    "ppe": "An toàn lao động",
    "fire": "Phòng cháy chữa cháy",
    "traffic": "Giao thông thông minh",
    "behavior": "Phân tích hành vi",
    "retail_analytics": "Phân tích Bán lẻ",
}


@dataclass(frozen=True)
class ModelSpec:
    """Mô tả một mô hình: tên hiển thị, chế độ, backend, lớp detector.

    ``detector`` là tên class trong package ``recognition`` sẽ khởi tạo khi chạy
    thật (dùng để factory nạp đúng detector; import lazy để không kéo torch).
    """

    key: str
    display_name: str
    mode: MonitoringMode
    detector: str
    open_vocab: bool = False
    notes: str = ""


# Danh mục mô hình lõi mà tầng nhận diện hỗ trợ.
MODEL_SPECS: Dict[str, ModelSpec] = {
    "yolo_nas": ModelSpec(
        key="yolo_nas",
        display_name="YOLO-NAS-S",
        mode=MonitoringMode.STANDARD,
        detector="YoloNasDetector",
        notes="Phát hiện từ vựng cố định: người, xe (car/motorcycle/truck/bus), đồ vật COCO.",
    ),
    "locate_anything": ModelSpec(
        key="locate_anything",
        display_name="LocateAnything-3B",
        mode=MonitoringMode.SMART,
        detector="LocateAnythingDetector",
        open_vocab=True,
        notes="Phát hiện theo mô tả ngôn ngữ tự nhiên (open-vocabulary).",
    ),
    "ppe_crop": ModelSpec(
        key="ppe_crop",
        display_name="YOLO-NAS + LocateAnything (Crop Mode)",
        mode=MonitoringMode.SMART,
        detector="PPEDetector",
        open_vocab=True,
        notes="YOLO-NAS cắt người → LocateAnything kiểm tra đồ bảo hộ trên từng crop.",
    ),
    "defect_inspector": ModelSpec(
        key="defect_inspector",
        display_name="So mẫu chuẩn (SSIM + CNN + OCR)",
        mode=MonitoringMode.DEFECT,
        detector="DefectInspector",
        notes="So khớp ảnh với mẫu chuẩn (golden sample) để phát hiện lỗi sản xuất.",
    ),
}


def resolve_model(mode: MonitoringMode, has_ppe: bool = False) -> ModelSpec:
    """Chọn ``ModelSpec`` mặc định cho một chế độ giám sát."""
    if mode == MonitoringMode.STANDARD:
        return MODEL_SPECS["yolo_nas"]
    if mode == MonitoringMode.DEFECT:
        return MODEL_SPECS["defect_inspector"]
    # SMART: PPE dùng hybrid crop, còn lại LocateAnything thuần.
    return MODEL_SPECS["ppe_crop"] if has_ppe else MODEL_SPECS["locate_anything"]


def spec_for(display_name: str) -> Optional[ModelSpec]:
    """Tra ``ModelSpec`` từ tên hiển thị (vd 'LocateAnything-3B')."""
    for spec in MODEL_SPECS.values():
        if spec.display_name == display_name:
            return spec
    return None
