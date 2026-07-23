"""Test bộ định tuyến model — backend suy luận cùng kết quả với frontend."""

from recognition.base import MonitoringMode
from recognition.router import infer_monitoring_config
from recognition.registry import resolve_model


def test_people_counting_routes_to_yolo_line():
    cfg = infer_monitoring_config("Đếm số người đi qua cổng vào ra")
    assert cfg.mode == MonitoringMode.STANDARD
    assert cfg.model == "YOLO-NAS-S"
    assert cfg.target == "person"
    assert cfg.counting_type == "line"
    assert cfg.rule == "cross_line"


def test_vehicle_counting_routes_to_yolo():
    cfg = infer_monitoring_config("đếm ô tô đi qua trạm")
    assert cfg.mode == MonitoringMode.STANDARD
    assert cfg.target == "vehicle"


def test_ppe_routes_to_hybrid_crop():
    cfg = infer_monitoring_config("phát hiện công nhân không đội mũ bảo hộ")
    assert cfg.mode == MonitoringMode.SMART
    assert cfg.model == "YOLO-NAS + LocateAnything (Crop Mode)"
    assert cfg.rule == "safety_violation"


def test_open_vocab_routes_to_locate_anything():
    cfg = infer_monitoring_config("đếm số hộp carton màu vàng trên kệ")
    assert cfg.mode == MonitoringMode.SMART
    assert cfg.model == "LocateAnything-3B"
    assert cfg.rule == "object_counting"
    assert cfg.search_query == "đếm số hộp carton màu vàng trên kệ"


def test_defect_routes_to_smart():
    cfg = infer_monitoring_config("phát hiện sản phẩm bị móp méo, xước")
    assert cfg.mode == MonitoringMode.SMART
    assert cfg.rule == "defect_detected"
    assert cfg.config["similarityThreshold"] == 0.82


def test_intrusion_known_target_is_standard():
    cfg = infer_monitoring_config("cảnh báo khi có người đi vào khu vực cấm")
    assert cfg.mode == MonitoringMode.STANDARD
    assert cfg.rule == "enter_area"


def test_roi_sets_scope():
    cfg = infer_monitoring_config("đếm người vào ra", has_roi=True)
    assert cfg.scope == "roi"
    cfg2 = infer_monitoring_config("đếm người vào ra", has_roi=False)
    assert cfg2.scope == "whole_scene"


def test_now_keyword_shrinks_cooldown():
    cfg = infer_monitoring_config("báo ngay khi người xâm nhập")
    assert cfg.config["cooldown"] == 15


def test_resolve_model_matches_router():
    cfg = infer_monitoring_config("đếm ô tô qua trạm")
    spec = resolve_model(cfg.mode, has_ppe=False)
    assert spec.display_name == "YOLO-NAS-S"

    cfg2 = infer_monitoring_config("người không đội mũ bảo hộ")
    spec2 = resolve_model(cfg2.mode, has_ppe=True)
    assert spec2.display_name == "YOLO-NAS + LocateAnything (Crop Mode)"
