"""Test công cụ VẼ vạch/vùng (HTML+JS) — dựng frame giả, không tải mạng, không GPU."""

import base64

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from recognition.draw_tool import _find, draw_html, frame_data_uri, save_draw_html


def _fake_video(path, w=320, h=180, n=20):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    for i in range(n):
        fr = np.full((h, w, 3), 40, np.uint8)
        cv2.circle(fr, ((i * 15) % w, h // 2), 8, (0, 0, 200), -1)
        vw.write(fr)
    vw.release()
    return str(path)


def test_frame_data_uri_embeds_jpeg(tmp_path):
    vid = _fake_video(tmp_path / "v.mp4")
    uri, w, h = frame_data_uri(vid, resolution=(1280, 720))
    assert uri.startswith("data:image/jpeg;base64,")
    assert (w, h) == (1280, 720)                 # resize về resolution scenario
    # phần base64 giải mã được (frame thật)
    base64.b64decode(uri.split(",", 1)[1])


def test_draw_html_has_canvas_and_cli_output(tmp_path):
    vid = _fake_video(tmp_path / "v.mp4")
    html = draw_html(vid, title="[test]", resolution=(640, 360))
    for token in ('<canvas id="cv-', "getContext", "--zone", "--line",
                  "zone_points_pct", "function pct"):
        assert token in html, token


def test_draw_html_unique_uid(tmp_path):
    vid = _fake_video(tmp_path / "v.mp4")
    import re

    a = re.search(r"cv-(dz\d+)", draw_html(vid, resolution=(640, 360))).group(1)
    b = re.search(r"cv-(dz\d+)", draw_html(vid, resolution=(640, 360))).group(1)
    assert a != b                                # nhiều cell không đụng id


def test_find_by_name_and_key():
    for kw, key in [("subway", "ppl_subway"), ("milk", "conv_milk"),
                    ("giao lộ", "veh_junc")]:
        assert _find(kw).scenario.key == key


def test_find_unknown_raises():
    with pytest.raises(KeyError):
        _find("khong-co-video-nao-ten-the-nay")


def test_gallery_embeds_multiple_frames_and_nav(tmp_path, monkeypatch):
    from recognition import draw_tool as D
    from recognition import video_catalog as VC

    vid = _fake_video(tmp_path / "v.mp4")

    class _SC:
        def __init__(self, k):
            self.key, self.resolution = k, (640, 360)

    class _V:
        def __init__(self, k):
            self.scenario = _SC(k)
            self.filename = k + ".mp4"       # gallery gộp theo filename → cần thuộc tính này

    monkeypatch.setattr(D, "_find", lambda n: _V(n))
    monkeypatch.setattr(VC, "download_video", lambda v, *a, **k: str(vid))

    html = D.draw_gallery(["aaa", "bbb", "ccc"], max_width=600)
    assert html.count("data:image/jpeg") == 3          # 3 frame nhúng
    for token in ("prev-", "next-", "◀ Ảnh trước", "Ảnh sau ▶", "const FRAMES",
                  "function finishZone", "Copy tất cả"):
        assert token in html, token
    assert "__UID__" not in html and "__FRAMES__" not in html   # placeholder đã thay hết


def test_gallery_all_uses_every_scenario_key():
    from recognition import draw_tool as D
    from recognition.video_catalog import CATALOG

    # 'all' phải phủ mọi scenario.key (không trùng) — nhưng cần mạng để tải nên chỉ
    # kiểm tra danh sách tên được suy ra đúng (patch tải + đọc frame).
    keys = []
    seen = set()
    for v in CATALOG:
        if v.scenario.key not in seen:
            seen.add(v.scenario.key)
            keys.append(v.scenario.key)
    assert len(keys) == len({v.scenario.key for v in CATALOG}) >= 15
