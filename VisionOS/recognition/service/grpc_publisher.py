# -*- coding: utf-8 -*-
"""gRPC Publisher — đẩy bbox (stream) + events (unary) về Backend.

Theo AI_SERVICE_CONTRACT.md mục 4:
  • PublishFrames: 1 stream duy nhất cho toàn tiến trình, dùng chung mọi job.
    Cho phép mất frame (Backend drop oldest khi queue 100 đầy).
  • PublishEvent: Unary, mỗi sự kiện 1 RPC, phải có ack. Retry backoff ≤ 5s.
  • Reconnect backoff tối đa 5s khi stream đứt, giữ nguyên job đang chạy.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from typing import List, Optional

__all__ = ["OverlayPublisher"]


class OverlayPublisher:
    """gRPC client đẩy kết quả AI sang Backend (OverlayIngest service).

    Tự động:
      * Chờ Backend sẵn sàng (Health RPC) khi khởi tạo.
      * Mở 1 stream PublishFrames dùng chung, gửi frame qua internal queue.
      * Retry + backoff khi stream đứt hoặc PublishEvent lỗi.
    """

    def __init__(self, target: Optional[str] = None, max_backoff: float = 5.0):
        self._target = target or os.environ.get("OVERLAY_GRPC_TARGET", "127.0.0.1:9090")
        self._max_backoff = max_backoff
        self._channel = None
        self._stub = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=200)
        self._stream_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._total_sent = 0
        self._total_dropped = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        """Kết nối gRPC channel và bắt đầu stream thread."""
        import grpc
        from .proto import overlay_pb2_grpc

        self._channel = grpc.insecure_channel(self._target)
        self._stub = overlay_pb2_grpc.OverlayIngestStub(self._channel)
        self._running = True

        # Chờ Backend sẵn sàng (non-blocking, chạy nền)
        t = threading.Thread(target=self._wait_backend_and_start_stream, daemon=True)
        t.start()
        print(f"[gRPC] Publisher khởi tạo → target={self._target}", flush=True)

    def stop(self):
        """Đóng stream + channel."""
        self._running = False
        # Đẩy sentinel để unblock queue.get()
        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=3.0)
        if self._channel:
            self._channel.close()
            self._channel = None
        print(f"[gRPC] Publisher dừng. Sent={self._total_sent}, Dropped={self._total_dropped}", flush=True)

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def publish_frame(self, camera_id: str, job_key: str,
                      captured_at_ms: int, width: int, height: int,
                      boxes: List[dict]):
        """Đẩy 1 OverlayFrame vào queue (non-blocking).

        boxes: list of {track_id, class_name, confidence, x, y, w, h} (pixel).
        Frame rỗng (boxes=[]) vẫn gửi để client xoá box cũ.
        """
        from .proto import overlay_pb2

        frame = overlay_pb2.OverlayFrame(
            camera_id=str(camera_id),
            job_key=str(job_key),
            captured_at_ms=int(captured_at_ms),
            resolution=overlay_pb2.Resolution(width=int(width), height=int(height)),
            boxes=[
                overlay_pb2.BoundingBox(
                    track_id=str(b.get("track_id", "")),
                    class_name=str(b.get("class_name", "")),
                    confidence=float(b.get("confidence", 0.0)),
                    x=float(b.get("x", 0.0)),
                    y=float(b.get("y", 0.0)),
                    w=float(b.get("w", 0.0)),
                    h=float(b.get("h", 0.0)),
                    velocity_x=float(b.get("velocity_x", 0.0)),
                    velocity_y=float(b.get("velocity_y", 0.0)),
                )
                for b in boxes
            ],
        )

        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame (backpressure)
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def publish_event(self, camera_id: str, job_key: str,
                      track_id: str, class_name: str,
                      kind: str, occurred_at_ms: int):
        """Gửi 1 DetectionEvent (unary, có retry + backoff, không được mất).

        kind: "START" | "IN" | "OUT"
        """
        if not self._stub:
            return
        from .proto import overlay_pb2

        kind_map = {
            "START": overlay_pb2.DetectionEvent.START,
            "start": overlay_pb2.DetectionEvent.START,
            "IN": overlay_pb2.DetectionEvent.IN,
            "in": overlay_pb2.DetectionEvent.IN,
            "OUT": overlay_pb2.DetectionEvent.OUT,
            "out": overlay_pb2.DetectionEvent.OUT,
        }
        kind_enum = kind_map.get(kind, overlay_pb2.DetectionEvent.START)

        event = overlay_pb2.DetectionEvent(
            event_id=uuid.uuid4().hex,
            camera_id=str(camera_id),
            job_key=str(job_key),
            track_id=str(track_id),
            class_name=str(class_name),
            kind=kind_enum,
            occurred_at_ms=int(occurred_at_ms),
        )

        # Retry with backoff (max 5s) — chạy trong thread riêng để không block
        threading.Thread(
            target=self._send_event_with_retry,
            args=(event,),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _wait_backend_and_start_stream(self):
        """Chờ Backend sẵn sàng rồi mở PublishFrames stream."""
        from .proto import overlay_pb2
        import grpc

        backoff = 0.5
        while self._running:
            try:
                resp = self._stub.Health(overlay_pb2.HealthRequest(), timeout=3.0)
                if resp.ready:
                    print(f"[gRPC] Backend sẵn sàng (Health OK)", flush=True)
                    break
            except grpc.RpcError as e:
                print(f"[gRPC] Chờ Backend... ({e.code().name})", flush=True)
            except Exception as e:
                print(f"[gRPC] Chờ Backend... ({e})", flush=True)
            time.sleep(min(backoff, self._max_backoff))
            backoff = min(backoff * 1.5, self._max_backoff)

        if not self._running:
            return

        self._connected = True
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    def _stream_loop(self):
        """Vòng lặp gửi OverlayFrame qua stream. Tự reconnect khi đứt."""
        import grpc

        while self._running:
            try:
                summary = self._stub.PublishFrames(self._frame_generator())
                # Stream kết thúc bình thường (server đóng)
                if summary:
                    self._total_dropped += summary.dropped
                    if summary.dropped > 0:
                        print(f"[gRPC] Stream kết thúc. Dropped={summary.dropped}", flush=True)
            except grpc.RpcError as e:
                if self._running:
                    print(f"[gRPC] Stream đứt ({e.code().name}). Reconnect...", flush=True)
            except Exception as e:
                if self._running:
                    print(f"[gRPC] Stream lỗi ({e}). Reconnect...", flush=True)

            if self._running:
                time.sleep(min(1.0, self._max_backoff))

    def _frame_generator(self):
        """Generator cho client-streaming PublishFrames RPC."""
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if frame is None:  # sentinel
                break
            self._total_sent += 1
            yield frame

    def _send_event_with_retry(self, event):
        """Gửi DetectionEvent với retry + exponential backoff."""
        import grpc

        backoff = 0.2
        max_attempts = 10
        for attempt in range(max_attempts):
            if not self._running:
                return
            try:
                ack = self._stub.PublishEvent(event, timeout=5.0)
                if ack.accepted:
                    return
            except grpc.RpcError:
                pass
            except Exception:
                pass
            time.sleep(min(backoff, self._max_backoff))
            backoff = min(backoff * 2, self._max_backoff)

        print(f"[gRPC] ⚠️ Event {event.event_id[:8]} không gửi được sau {max_attempts} lần", flush=True)
