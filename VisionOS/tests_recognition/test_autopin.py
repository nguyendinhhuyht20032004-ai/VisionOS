"""Test guard logic của ``_autopin_transformers`` — KHÔNG chạy pip/exec thật.

Chỉ kiểm chứng phần quyết định (đã thử chưa / version có khớp không). Nhánh cài
đặt + re-exec có side-effect (pip, os.execv) nên ta luôn mock ``os.execv`` để chắc
chắn test không bao giờ tự thay tiến trình.
"""

import importlib.metadata as md
import os

import run_eval


def _block_execv(monkeypatch):
    """Chặn os.execv; trả về dict cờ để khẳng định KHÔNG exec."""
    flag = {"exec": False}
    monkeypatch.setattr(os, "execv", lambda *a, **k: flag.__setitem__("exec", True))
    return flag


def test_no_reexec_when_already_tried(monkeypatch):
    # _LA_PIN_TRIED=1 (đã thử 1 lần) → PHẢI return, tuyệt đối không exec lần 2,
    # kể cả khi version vẫn sai (tránh vòng lặp cài-exec vô hạn).
    monkeypatch.setenv("_LA_PIN_TRIED", "1")
    flag = _block_execv(monkeypatch)
    monkeypatch.setattr(md, "version", lambda name: "5.0.0")   # vẫn sai
    run_eval._autopin_transformers(target="4.57.1")
    assert flag["exec"] is False


def test_noop_when_version_matches(monkeypatch):
    # version đã khớp target → return ngay, không exec, không cài.
    monkeypatch.delenv("_LA_PIN_TRIED", raising=False)
    flag = _block_execv(monkeypatch)
    monkeypatch.setattr(md, "version", lambda name: "4.57.1")
    run_eval._autopin_transformers(target="4.57.1")
    assert flag["exec"] is False
