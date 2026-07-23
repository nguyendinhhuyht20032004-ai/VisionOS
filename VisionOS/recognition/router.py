"""Bộ định tuyến mô hình — từ mô tả ngôn ngữ tự nhiên → chọn model + cấu hình.

Đây là **bộ não** của trải nghiệm "mô tả bằng lời, hệ thống tự chọn mô hình" của
sản phẩm. Nó port trung thực hàm ``inferMonitoringConfig`` trong
``src/components/PipelineBuilder.tsx`` sang Python để backend suy luận cùng một
kết quả với frontend:

  * mô tả có đối tượng *đã biết* (người/xe) và không cần open-vocab
    → chế độ **STANDARD** dùng ``YOLO-NAS-S``;
  * mô tả PPE/đồ bảo hộ
    → **SMART** dùng hybrid ``YOLO-NAS + LocateAnything (Crop Mode)``;
  * còn lại (mô tả tự do, lỗi sản phẩm, đồ bị bỏ lại/lấy đi…)
    → **SMART** dùng ``LocateAnything-3B``.

Toàn bộ là logic thuần Python (chỉ so khớp từ khoá) nên unit-test được dễ dàng và
đảm bảo backend/frontend không lệch nhau.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import MonitoringMode

__all__ = ["MonitoringConfig", "infer_monitoring_config", "KEYWORDS"]

# Bộ từ khoá — giữ đúng như bản TS để hai phía suy luận giống hệt nhau.
KEYWORDS: Dict[str, List[str]] = {
    "ppe": ["mũ", "áo phản quang", "bảo hộ", "ppe", "an toàn"],
    "vehicle": ["xe", "ô tô", "oto", "xe máy", "motorcycle", "truck", "tải"],
    "person": ["người", "khách", "nhân viên", "công nhân", "person"],
    "count": ["đếm", "số lượng", "bao nhiêu", "count"],
    "line": ["vào ra", "ra vào", "đi qua", "qua cổng", "cross", "line"],
    "exit": ["rời khỏi", "đi ra", "exit"],
    "intrusion": ["xâm nhập", "đi vào", "vào khu vực", "enter"],
    "loiter": ["lảng vảng", "ở lại lâu", "loiter", "quá lâu"],
    "defect": ["lỗi", "móp", "rách", "xước", "hỏng", "defect"],
    "abandoned": ["bỏ lại", "leaving", "balo", "ba lô", "túi"],
    "removal": ["lấy hàng", "lấy khỏi", "remove"],
    "now": ["ngay"],
}


def _has(lw: str, group: str) -> bool:
    return any(kw in lw for kw in KEYWORDS[group])


@dataclass
class MonitoringConfig:
    """Kết quả định tuyến — đủ để khởi tạo detector + rule engine + pipeline."""

    mode: MonitoringMode
    model: str
    rule: str
    scope: str                       # "roi" | "whole_scene"
    counting_type: str               # "line" | "zone"
    target: Optional[str] = None     # "person" | "vehicle" (chỉ ở STANDARD)
    search_query: Optional[str] = None  # cụm mô tả cho LocateAnything (SMART)
    config: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "model": self.model,
            "rule": self.rule,
            "scope": self.scope,
            "countingType": self.counting_type,
            "target": self.target,
            "searchQuery": self.search_query,
            "config": self.config,
        }


def infer_monitoring_config(
    text: str,
    has_roi: bool = False,
    max_limit: int = 5,
) -> MonitoringConfig:
    """Suy luận model + cấu hình từ mô tả ``text``.

    ``has_roi``  : người dùng đã khoanh vùng/vạch chưa (quyết định scope).
    ``max_limit``: ngưỡng đếm mặc định khi mô tả yêu cầu "số lượng" (khớp state
                   ``maxLimit`` ở frontend).
    """
    lw = text.lower()
    has_ppe = _has(lw, "ppe")
    has_vehicle = _has(lw, "vehicle")
    has_person = _has(lw, "person")
    asks_count = _has(lw, "count")
    uses_line = _has(lw, "line")
    uses_exit = _has(lw, "exit")
    uses_intrusion = _has(lw, "intrusion")
    uses_loiter = _has(lw, "loiter")
    has_defect = _has(lw, "defect")
    has_abandoned = _has(lw, "abandoned")
    has_removal = _has(lw, "removal")
    now = _has(lw, "now")

    uses_known_target = has_person or has_vehicle
    needs_open = (
        has_ppe
        or has_defect
        or has_abandoned
        or has_removal
        or (not uses_known_target and len(text.strip()) > 0)
    )

    # ---- Nhánh STANDARD (YOLO-NAS, từ vựng cố định) ----------------------- #
    if not needs_open and uses_known_target:
        target = "vehicle" if has_vehicle else "person"
        if uses_line:
            rule = "cross_line"
        elif uses_exit:
            rule = "exit_area"
        elif uses_intrusion:
            rule = "enter_area"
        elif uses_loiter:
            rule = "loitering"
        elif asks_count:
            rule = "object_counting"
        else:
            rule = "appear"
        return MonitoringConfig(
            mode=MonitoringMode.STANDARD,
            model="YOLO-NAS-S",
            target=target,
            rule=rule,
            scope="roi" if has_roi else "whole_scene",
            counting_type="line" if (uses_line or rule == "cross_line") else "zone",
            config={
                "alertDuration": 60 if uses_loiter else 10,
                "alertCount": max_limit if asks_count else 1,
                "cooldown": 15 if now else 60,
                "confidence": 0.65,
                "iou": 0.45,
                "tracker": "bytetrack",
                "frameSkip": 0 if uses_line else 1,
                "inferenceFps": 15,
            },
        )

    # ---- Nhánh SMART (LocateAnything / hybrid Crop) ---------------------- #
    if has_ppe:
        rule = "safety_violation"
    elif has_defect:
        rule = "defect_detected"
    elif has_abandoned:
        rule = "abandoned_object"
    elif has_removal:
        rule = "object_removed"
    elif uses_intrusion:
        rule = "enter_area"
    elif asks_count:
        rule = "object_counting"
    else:
        rule = "semantic_match"

    return MonitoringConfig(
        mode=MonitoringMode.SMART,
        model="YOLO-NAS + LocateAnything (Crop Mode)" if has_ppe else "LocateAnything-3B",
        rule=rule,
        scope="roi" if has_roi else "whole_scene",
        counting_type="zone",
        search_query=(
            "người không đội mũ bảo hộ, người không mặc áo phản quang"
            if has_ppe
            else text
        ),
        config={
            "similarityThreshold": 0.82 if has_defect else 0.78,
            "retrievalTopK": 5 if has_roi else 8,
            "cooldown": 15 if now else 60,
            "alertDuration": 60 if uses_loiter else 10,
            "alertCount": max_limit if asks_count else 1,
        },
    )
