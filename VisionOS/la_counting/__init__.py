"""la_counting — thư viện đếm đối tượng dựa trên mô hình LocateAnything-3B.

Package này tách phần logic *thuần* (parse toạ độ, cấu hình bài toán, pipeline
đếm bằng supervision) ra khỏi notebook Kaggle để có thể:

  * import và **unit-test** mà KHÔNG cần GPU / torch / transformers, và
  * tái sử dụng lại y hệt trong notebook chạy trên GPU (benchmark mô hình thật).

Các module:
  - ``parsing``   : ``Detection`` + hàm parse text của model ra bbox (thuần Python).
  - ``scenarios`` : định nghĩa 3 bài toán đếm (người ra/vào, băng chuyền, xe cộ).
  - ``counting``  : ``CountingPipeline`` bọc ByteTrack + LineZone của supervision.
  - ``detector``  : ``LocateAnythingDetector`` (import torch/transformers *lazy*).

Import ``detector`` chỉ cần thiết khi chạy mô hình thật; mọi thứ còn lại import
được ở môi trường chỉ có numpy + supervision.
"""

from .parsing import Detection, parse_boxes, rescale_box
from .scenarios import (
    LineConfig,
    Scenario,
    SCENARIOS,
    PEOPLE_IN_OUT,
    CONVEYOR,
    VEHICLES,
    build_line_zone,
)
from .counting import CountingPipeline, CountResult

__all__ = [
    "Detection",
    "parse_boxes",
    "rescale_box",
    "LineConfig",
    "Scenario",
    "SCENARIOS",
    "PEOPLE_IN_OUT",
    "CONVEYOR",
    "VEHICLES",
    "build_line_zone",
    "CountingPipeline",
    "CountResult",
]
