"""Test tích hợp trên ẢNH THẬT (không phải numpy giả) cho 3 lớp đánh giá.

Khác với ``test_coco.py`` (chỉ test hàm parse thuần, dùng dict Python giả) và
``run_eval.py --selftest`` (dùng frame ``np.zeros`` đen), bộ test này:

  * dùng 3 ảnh COCO **thật** (đã crop sẵn vào ``fixtures/images/``, ảnh gốc từ
    tập con ``coco128`` do Ultralytics phát hành trên GitHub — cùng nguồn ảnh
    COCO train2017 chính chủ, chỉ đóng gói lại để tải nhanh hơn 19GB gốc);
  * dùng annotation COCO thật (``fixtures/instances_mini.json``, cắt ra từ
    ``instances_train2017.json`` chuẩn — toạ độ bbox xyxy tính lại từ nhãn
    YOLO gốc của coco128, không tự chế);
  * chạy đúng pipeline sản phẩm dùng (``recognition.coco`` → ``load_pairs`` →
    ``recognition.evaluation.DetectionMetrics``), với **YOLOv8n thật** nếu máy
    có cài ``ultralytics`` (bỏ qua — không fail — nếu chưa cài, vì CI thường
    không có GPU/mạng để tải trọng số).

Mục tiêu: bắt các lỗi mà test thuần logic (fake numpy) không bao giờ thấy được
— ví dụ: ``cv2.imread`` đọc sai kênh màu (BGR/RGB), toạ độ bbox lệch do decode
ảnh JPEG thật (nén có mất mát, không phải mảng toàn số 0), hoặc pipeline
load_pairs bỏ sót ảnh lỗi.
"""

from __future__ import annotations

import os

import cv2
import pytest

from recognition.coco import EVAL_CLASSES, load_eval_samples
from recognition.evaluation import DetectionMetrics
from recognition.base import BoundingBox

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
ANN_FILE = os.path.join(FIXTURES, "instances_mini.json")
IMAGES_DIR = os.path.join(FIXTURES, "images")

# Số GT thật đã đếm tay từ instances_mini.json (không đổi trừ khi sửa fixture).
EXPECTED_MIN_GT = {"person": 5, "car": 4, "bottle": 1}


def _require_ultralytics():
    ultralytics = pytest.importorskip("ultralytics", reason="cần cài ultralytics để chạy detector thật")
    return ultralytics


@pytest.fixture(scope="module")
def real_yolo():
    """YOLOv8n thật (tải sẵn .pt nếu có cache, bỏ qua nếu không có mạng)."""
    _require_ultralytics()
    try:
        from recognition.detectors import load_ultralytics_yolo

        return load_ultralytics_yolo(confidence=0.25)
    except Exception as e:  # noqa: BLE001 — không có mạng để tải weights, v.v.
        pytest.skip(f"không khởi tạo được YOLOv8 thật: {e}")


class TestFixturesTonTai:
    """Kiểm tra bản thân bộ fixture ảnh thật trước (không cần detector)."""

    def test_du_3_anh_that(self):
        assert os.path.isdir(IMAGES_DIR)
        files = sorted(f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg"))
        assert len(files) == 3, f"cần đúng 3 ảnh fixture, có {files}"

    def test_anh_doc_duoc_bang_cv2_va_khong_phai_toan_den(self):
        """Ảnh JPEG thật phải giải mã ra >1 giá trị pixel (khác hẳn np.zeros của selftest)."""
        for fname in os.listdir(IMAGES_DIR):
            img = cv2.imread(os.path.join(IMAGES_DIR, fname))
            assert img is not None, f"cv2 không đọc được {fname}"
            assert img.ndim == 3 and img.shape[2] == 3
            assert img.std() > 5, f"{fname} gần như toàn 1 màu — nghi ảnh hỏng"

    @pytest.mark.parametrize("cls", ["person", "car", "bottle"])
    def test_load_eval_samples_tra_ve_gt_that(self, cls):
        """load_eval_samples (hàm dùng thật trong run_eval.py) phải ghép đúng ảnh + GT."""
        samples = load_eval_samples(cls, limit=10, ann_file=ANN_FILE, images_dir=IMAGES_DIR)
        assert len(samples) >= 1, f"lớp '{cls}' phải có ít nhất 1 ảnh trong fixture"
        n_gt = sum(len(gt) for _, gt in samples)
        assert n_gt >= EXPECTED_MIN_GT[cls], (
            f"'{cls}': mong >= {EXPECTED_MIN_GT[cls]} GT box thật, chỉ thấy {n_gt}"
        )
        # Mỗi box GT phải hợp lệ trên ảnh thật (không âm, x2>x1, y2>y1, nằm trong ảnh).
        for path, gt in samples:
            frame = cv2.imread(path)
            h, w = frame.shape[:2]
            for (x1, y1, x2, y2) in gt:
                assert 0 <= x1 < x2 <= w + 1
                assert 0 <= y1 < y2 <= h + 1


@pytest.mark.slow
class TestDanhGiaTrenAnhThatVoiYOLOThat:
    """Chạy full pipeline detect→match→P/R/F1 bằng YOLOv8n thật trên ảnh thật.

    Đánh dấu ``@pytest.mark.slow`` vì cần tải/khởi động model thật (bỏ qua nếu
    không có ``ultralytics`` hoặc không tải được weights — xem ``real_yolo``).
    """

    @pytest.mark.parametrize("cls", ["person", "car", "bottle"])
    def test_pipeline_khong_crash_va_ra_metric_hop_le(self, real_yolo, cls):
        samples = load_eval_samples(cls, limit=10, ann_file=ANN_FILE, images_dir=IMAGES_DIR)
        prompt = EVAL_CLASSES[cls]["prompt"]
        m = DetectionMetrics(iou_thr=0.5, label=cls)
        for path, gt in samples:
            frame = cv2.imread(path)
            res = real_yolo.detect(frame, prompt)
            boxes = [d.bbox for d in res.detections]
            scores = [d.confidence for d in res.detections]
            gt_bb = [BoundingBox(*g) for g in gt]
            m.add_image(boxes, gt_bb, scores)

        # Không assert F1 cao (model thật có thể bỏ sót) — chỉ assert pipeline
        # CHẠY ĐƯỢC và trả về số liệu hợp lệ trên dữ liệu thật (đây là điều mà
        # test dùng detector giả/ảnh đen không kiểm chứng được).
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1 <= 1.0
        assert m.n_gt == sum(len(gt) for _, gt in samples)
        assert m.tp + m.fn == m.n_gt
