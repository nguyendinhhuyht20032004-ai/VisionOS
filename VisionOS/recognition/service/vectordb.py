"""Vector database cho service — lưu "ngoại hình" vật đã đếm + tra cứu (ReID / tìm kiếm).

Dùng **Qdrant** khi có (``QDRANT_URL``); nếu không kết nối được thì tự rơi về **bộ nhớ
trong** (in-memory) để service vẫn chạy — phần đếm KHÔNG phụ thuộc vector DB.

Embedding (đặc trưng ngoại hình) tính bằng **histogram màu HSV** của ảnh crop — nhẹ, không
cần model phụ, đủ để demo ReID/tìm-kiếm; có thể thay bằng model ReID thật (osnet…) sau.
``numpy`` là phụ thuộc duy nhất bắt buộc; ``qdrant_client``/``cv2`` import lazy.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

EMBED_DIM = 256          # 16 bins Hue × 16 bins Saturation
_COLLECTION = "tracks"

__all__ = ["embed_crop", "VectorStore"]


def embed_crop(crop_bgr) -> List[float]:
    """Ảnh crop BGR → vector đặc trưng (hist màu HSV, L2-normalize, dài EMBED_DIM)."""
    import cv2
    import numpy as np

    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return [0.0] * EMBED_DIM
    hsv = cv2.cvtColor(cv2.resize(crop_bgr, (64, 64)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    v = hist.flatten().astype("float32")
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v.tolist()


class VectorStore:
    """Kho vector: Qdrant nếu có, không thì in-memory. API đồng nhất cho cả hai."""

    def __init__(self, url: Optional[str] = None, collection: str = _COLLECTION):
        self.url = url or os.environ.get("QDRANT_URL")
        self.collection = collection
        self.backend = "memory"
        self._client = None
        self._mem: List[dict] = []          # fallback: [{id,vector,payload}]
        self._n = 0
        if self.url:
            self._try_qdrant()

    # ------------------------------------------------------------------ #
    def _try_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(url=self.url, timeout=5.0)
            existing = {c.name for c in self._client.get_collections().collections}
            if self.collection not in existing:
                self._client.create_collection(
                    self.collection,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
            self.backend = "qdrant"
        except Exception as e:  # noqa: BLE001 — không có qdrant/không kết nối → dùng memory
            print(f"ℹ️  Vector DB: không dùng được Qdrant ({type(e).__name__}) → in-memory.")
            self._client = None
            self.backend = "memory"

    # ------------------------------------------------------------------ #
    def add_event(self, vector: List[float], payload: dict) -> int:
        """Lưu 1 sự kiện (vật đã đếm) với vector ngoại hình + payload (lớp, camera, thời gian…)."""
        pid = self._n
        self._n += 1
        if self.backend == "qdrant" and self._client is not None:
            from qdrant_client.models import PointStruct

            try:
                self._client.upsert(self.collection,
                                    [PointStruct(id=pid, vector=vector, payload=payload)])
            except Exception:  # noqa: BLE001 — lỗi mạng giữa chừng → rơi về memory cho bền
                self.backend = "memory"
                self._mem.append({"id": pid, "vector": vector, "payload": payload})
        else:
            self._mem.append({"id": pid, "vector": vector, "payload": payload})
        return pid

    def search(self, vector: List[float], limit: int = 5) -> List[dict]:
        """Tìm ``limit`` vật GIỐNG NHẤT (cosine) → [{id, score, payload}]."""
        if self.backend == "qdrant" and self._client is not None:
            try:
                res = self._client.search(self.collection, query_vector=vector, limit=limit)
                return [{"id": r.id, "score": float(r.score), "payload": r.payload or {}} for r in res]
            except Exception:  # noqa: BLE001
                pass
        import numpy as np

        if not self._mem:
            return []
        q = np.asarray(vector, dtype="float32")
        qn = np.linalg.norm(q) or 1.0
        scored = []
        for it in self._mem:
            v = np.asarray(it["vector"], dtype="float32")
            vn = np.linalg.norm(v) or 1.0
            scored.append((float(q @ v / (qn * vn)), it))
        scored.sort(key=lambda x: -x[0])
        return [{"id": it["id"], "score": s, "payload": it["payload"]} for s, it in scored[:limit]]

    def recent(self, limit: int = 20) -> List[dict]:
        """Sự kiện GẦN NHẤT (mới → cũ)."""
        if self.backend == "qdrant" and self._client is not None:
            try:
                pts, _ = self._client.scroll(self.collection, limit=limit, with_payload=True,
                                             order_by=None)
                items = [{"id": p.id, "payload": p.payload or {}} for p in pts]
                return sorted(items, key=lambda x: x["payload"].get("ts", 0), reverse=True)[:limit]
            except Exception:  # noqa: BLE001
                pass
        return [{"id": it["id"], "payload": it["payload"]} for it in self._mem[-limit:]][::-1]

    def count(self) -> int:
        if self.backend == "qdrant" and self._client is not None:
            try:
                return int(self._client.count(self.collection).count)
            except Exception:  # noqa: BLE001
                pass
        return len(self._mem)

    def status(self) -> dict:
        return {"backend": self.backend, "url": self.url, "collection": self.collection,
                "events": self.count()}


def make_event_payload(track_id: int, class_name: str, source: str, kind: str, image_base64: str = "", full_frame_base64: str = "", video_url: str = "") -> dict:
    """Payload chuẩn cho 1 sự kiện đếm (ts thêm ở nơi gọi để test được không phụ thuộc thời gian)."""
    return {"track_id": int(track_id), "class_name": str(class_name),
            "source": str(source), "counting_type": str(kind), "ts": time.time(),
            "image_base64": image_base64, "full_frame_base64": full_frame_base64, "video_url": video_url}
