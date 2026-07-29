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
