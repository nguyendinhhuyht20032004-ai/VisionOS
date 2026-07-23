"""recognition — tầng mô hình nhận diện của VisionOS, tập trung vào bài toán ĐẾM.

Package này xây dựng lại (rebuild) phần mô hình nhận diện mà sản phẩm CV_product
mô tả ở frontend, ưu tiên **các bài toán đếm** để kiểm thử hiệu quả trước:

    detector (YOLO-NAS / LocateAnything / giả lập)
        → tracking (CentroidTracker, giữ danh tính qua frame)
        → counting (đếm cắt vạch / đếm chiếm vùng)

Thiết kế "detector duck-typed" giúp toàn bộ luồng detect → track → đếm **chạy và
test được không cần GPU** (dùng ``ScriptedDetector`` với kịch bản tất định), rồi
thay bằng mô hình thật khi lên GPU mà không phải sửa pipeline.

Các module:
  - ``base``      : Detection / BoundingBox / DetectorResult / MonitoringMode (thuần).
  - ``geometry``  : point-in-polygon, cắt vạch (thuần, unit-test kỹ).
  - ``zones``     : Zone (đa giác) + CountingLine (vạch hai chiều).
  - ``tracking``  : CentroidTracker giữ danh tính (thuần Python).
  - ``scenarios`` : 4 bài toán đếm chuẩn (người/xe/kiện hàng/hàng chờ).
  - ``counting``  : CountingPipeline + CountResult (scorecard).
  - ``router``    : suy luận model từ mô tả ngôn ngữ tự nhiên (port từ frontend).
  - ``registry``  : sổ đăng ký mô hình ↔ nghiệp vụ.
  - ``detectors`` : YOLO-NAS / LocateAnything (lazy) + ScriptedDetector (test).
"""

from .base import BoundingBox, Detection, DetectorResult, MonitoringMode
from .geometry import line_crossing, point_in_polygon
from .zones import CountingLine, Zone
from .tracking import CentroidTracker, Track
from .scenarios import (
    CountScenario,
    COUNT_SCENARIOS,
    PEOPLE_IN_OUT,
    VEHICLES,
    PACKAGES,
    QUEUE,
)
from .counting import CountingPipeline, CountResult
from .router import MonitoringConfig, infer_monitoring_config
from .registry import ModelSpec, MODEL_SPECS, resolve_model

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectorResult",
    "MonitoringMode",
    "line_crossing",
    "point_in_polygon",
    "CountingLine",
    "Zone",
    "CentroidTracker",
    "Track",
    "CountScenario",
    "COUNT_SCENARIOS",
    "PEOPLE_IN_OUT",
    "VEHICLES",
    "PACKAGES",
    "QUEUE",
    "CountingPipeline",
    "CountResult",
    "MonitoringConfig",
    "infer_monitoring_config",
    "ModelSpec",
    "MODEL_SPECS",
    "resolve_model",
]
