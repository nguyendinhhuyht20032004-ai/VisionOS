"""Ứng dụng FastAPI cho AI service đếm theo camera (YOLO + supervision + vector DB).

Điểm chính:
  • Người dùng LẤY 1 FRAME (``/api/snapshot``) rồi VẼ vạch/vùng trên canvas → tạo job đếm.
  • Đếm bằng supervision (LineZone/PolygonZone) trên track ByteTrack; annotate màu theo lớp.
  • Vector DB (Qdrant/in-memory) lưu "ngoại hình" vật đã thấy → tìm kiếm / ReID.

Endpoint:
  GET  /                    — trang web: nhập nguồn + CANVAS vẽ vạch/vùng + xem đếm.
  GET  /healthz             — health check (Docker/K8s).
  GET  /api/snapshot        — lấy 1 frame (JPEG) từ nguồn để vẽ.
  POST /api/jobs            — tạo job đếm (kèm vạch/vùng người dùng vẽ).
  GET  /api/jobs[/{id}]     — liệt kê / số đếm.
  GET  /api/jobs/{id}/frame.jpg · /mjpeg — frame annotate / luồng MJPEG.
  POST /api/jobs/{id}/stop  — dừng job.
  GET  /api/events          — sự kiện gần nhất (vector DB).
  POST /api/search/similar  — upload ảnh → tìm vật giống (ReID).
  GET  /api/vectordb        — trạng thái vector DB.

Chạy: ``python run_service.py --host 0.0.0.0 --port 8000``.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .builder import get_detector, make_scenario
from .camera import FrameSource, encode_jpeg, grab_snapshot
from .engine import StreamingCounter
from .vectordb import VectorStore, embed_crop, make_event_payload

app = FastAPI(title="VisionOS · AI Counting Service")

os.makedirs("/data/videos", exist_ok=True)
app.mount("/api/videos", StaticFiles(directory="/data/videos"), name="videos")

_JOBS: "Dict[str, Job]" = {}
_LOCK = threading.Lock()
_VDB = VectorStore()          # Qdrant nếu có QDRANT_URL, không thì in-memory


def _save_event_video(frames, out_path, fps: float = 20.0):
    """Ghi list frame BGR ra MP4 **H.264** để TRÌNH DUYỆT phát được trong thẻ ``<video>``.

    OpenCV chỉ ghi tin cậy codec ``mp4v`` (MPEG-4 Part 2) mà Chrome/Firefox KHÔNG giải mã
    được trong ``<video>`` — đó là lý do clip trong "Lịch sử sự kiện" mở ra không xem được.
    Cách chắc ăn: ghi tạm bằng mp4v rồi transcode sang H.264 (yuv420p + faststart để phát
    và tua được trên web) bằng ffmpeg (đã cài sẵn trong image). Thiếu/lỗi ffmpeg thì giữ
    nguyên file mp4v (còn hơn không có video).
    """
    import subprocess

    import cv2

    if not frames:
        return
    h, w = frames[0].shape[:2]
    scale = 640 / w if w > 640 else 1.0
    W, H = int(w * scale), int(h * scale)
    tmp = out_path + ".mp4v.mp4"                       # file trung gian (codec mp4v)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    for f in frames:
        vw.write(cv2.resize(f, (W, H)) if scale != 1.0 else f)
    vw.release()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", out_path],
            check=True, timeout=120,
        )
        os.remove(tmp)
        print(f"✅ Video sự kiện (H.264): {out_path} · {len(frames)} frame")
    except Exception as e:  # noqa: BLE001 — thiếu/lỗi ffmpeg → dùng luôn mp4v
        os.replace(tmp, out_path)
        print(f"⚠️  ffmpeg lỗi ({e}) → giữ mp4v (một số trình duyệt vẫn phát): {out_path}")


class JobRequest(BaseModel):
    source: str
    prompt: str = "person"
    counting_type: str = "line"                  # "line" | "zone" | "fullscreen"
    line: Optional[List[float]] = None           # [x1,y1,x2,y2] % (người dùng vẽ)
    zone: Optional[List[List[float]]] = None     # [[x,y],…] % (người dùng vẽ)
    model: str = "yolo"                          # người/xe → YOLO
    resolution: List[int] = [960, 540]
    max_fps: float = 12.0                         # trần fps hiển thị (thực tế bị giới hạn bởi tốc độ YOLO)
    confidence: float = 0.25
    detect_every: int = 1                         # chạy YOLO mỗi N frame (>1 = nhanh hơn trên CPU)
    group_label: bool = False                     # True = gộp mọi loại xe → 1 nhãn "vehicle";
                                                  # False (mặc định) = GIỮ phân loại car/truck/bus (mỗi loại 1 màu)
    in_label: str = "IN"
    out_label: str = "OUT"
    anchor: Optional[str] = None
    record_events: bool = True                   # lưu ngoại hình vật vào vector DB


class Job:
    def __init__(self, job_id: str, req: JobRequest):
        self.id = job_id
        self.req = req
        self.scenario, self.kind = make_scenario(
            req.prompt, req.counting_type, req.line, req.zone,
            tuple(req.resolution), req.in_label, req.out_label, req.anchor, req.model)
        # Prompt là NHÓM nhiều lớp (vd "vehicle"→car/moto/truck/bus) + group_label → gộp nhãn.
        from ..coco import COCO_ALIASES
        p = req.prompt.lower().strip()
        self.merge_label = req.prompt if (req.group_label and len(COCO_ALIASES.get(p, [])) > 1) else None
        self.source = req.source
        self.fs: Optional[FrameSource] = None
        self.counter: Optional[StreamingCounter] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.status = "khởi tạo"
        self.error: Optional[str] = None
        self.last_jpg: Optional[bytes] = None
        self._stats: dict = {}
        self._recorded: set = set()          # track-id đã lưu vào vector DB
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        import cv2

        try:
            self.status = "đang nạp model (" + self.kind + ")"
            detector = get_detector(self.kind, self.req.confidence)
            # smoother_len NHỎ (2) → box BÁM SÁT vật (length lớn = trung bình nhiều frame → box
            # trễ, "chạy sau" vật nhanh). Chỉnh qua env SMOOTHER_LEN (1 = tắt, bám sát nhất).
            self.counter = StreamingCounter(self.scenario, detector,
                                            resolution=self.scenario.resolution,
                                            detect_every=self.req.detect_every,
                                            smoother_len=int(os.environ.get("SMOOTHER_LEN", "2")),
                                            merge_label=self.merge_label)
            self.status = "đang kết nối camera"
            self.fs = FrameSource(self.source).start()
            self.status = "đang chạy"
            interval = 1.0 / self.req.max_fps if self.req.max_fps > 0 else 0.0
            last = 0.0
            import collections
            import threading
            self._video_buffer = collections.deque(maxlen=40)  # ~ 1.5 seconds past
            self._record_video_frames = 0
            self._current_video_filename = ""
            self._video_frames_to_write = []
            
            while self.running:
                fr = self.fs.read()
                if fr is None:
                    if self.fs.error and not self.fs.alive:
                        self.error = self.fs.error
                        break
                    time.sleep(0.02)
                    continue
                # ĐỒNG BỘ: detect + vẽ NGAY trên frame này → box LUÔN ÔM SÁT vật (không trễ).
                # Throttle theo max_fps để đỡ tải; reader đã phát đúng FPS gốc (file) nên không tua nhanh.
                now = time.time()
                if interval and now - last < interval:
                    time.sleep(min(0.02, interval - (now - last)))
                    continue
                last = now
                out = self.counter.process(fr)
                
                # Push annotated frame to ring buffer
                self._video_buffer.append(out.copy())
                
                # If we are recording future frames
                if self._record_video_frames > 0:
                    self._video_frames_to_write.append(out.copy())
                    self._record_video_frames -= 1
                    
                    if self._record_video_frames == 0:
                        # Ghi clip ở LUỒNG NỀN (transcode H.264 hơi tốn thời gian) để không nghẽn vòng đếm.
                        threading.Thread(
                            target=_save_event_video,
                            args=(self._video_frames_to_write, self._current_video_filename),
                            daemon=True,
                        ).start()

                if self.req.record_events:
                    self._record_events()
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self.last_jpg = buf.tobytes()
                        self._stats = self.counter.stats()
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            # Job dừng giữa lúc đang gom frame → vẫn lưu nốt clip dở (đồng bộ, lúc tắt).
            if getattr(self, '_record_video_frames', 0) > 0 and getattr(self, '_video_frames_to_write', []):
                _save_event_video(self._video_frames_to_write, self._current_video_filename)

            self.running = False
            self.status = "lỗi" if self.error else "đã dừng"
            if self.fs is not None:
                self.fs.stop()

    def _record_events(self):
        """Lưu NGOẠI HÌNH mỗi track MỚI (1 lần/track) vào vector DB → tìm kiếm/ReID."""
        c = self.counter
        if c is None or c.last_det is None or c.last_frame is None:
            return
        det = c.last_det
        if det.tracker_id is None:
            return
        names = det.data.get("class_name") if getattr(det, "data", None) else None
        
        # Lấy danh sách track hợp lệ cần lưu
        valid_tracks = []
        for i, tid in enumerate(det.tracker_id):
            if tid is not None and int(tid) not in self._recorded:
                valid_tracks.append((i, int(tid)))
        
        if not valid_tracks:
            return
            
        # Nén ảnh toàn cảnh (full frame) một lần cho cả batch
        import cv2
        import base64
        full_frame_b64 = ""
        try:
            h, w = c.last_frame.shape[:2]
            scale = 640 / w if w > 640 else 1.0
            resized = cv2.resize(c.last_frame, (int(w*scale), int(h*scale)))
            succ, buf = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if succ:
                full_frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
        except Exception:
            pass

        # --- VIDEO TRIGGER LOGIC ---
        import time
        video_url = ""
        if getattr(self, '_record_video_frames', 0) == 0:
            os.makedirs("/data/videos", exist_ok=True)
            vid_id = int(time.time() * 1000)
            self._current_video_filename = f"/data/videos/event_{vid_id}.mp4"
            video_url = f"/api/videos/event_{vid_id}.mp4"
            self._video_frames_to_write = list(getattr(self, '_video_buffer', []))
            self._record_video_frames = 40  # ~ 2 seconds future
        else:
            video_url = self._current_video_filename.replace("/data/videos/", "/api/videos/")

        for i, tid in valid_tracks:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy[i])
            crop = c.last_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if getattr(crop, "size", 0) == 0:
                continue
            self._recorded.add(tid)
            nm = str(names[i]) if names is not None else self.req.prompt
            
            b64_str = ""
            success, buffer = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if success:
                b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

            try:
                _VDB.add_event(embed_crop(crop),
                               make_event_payload(tid, nm, self.source, self.scenario.counting_type, b64_str, full_frame_b64, video_url))
            except Exception:  # noqa: BLE001 — vector DB lỗi KHÔNG được làm hỏng việc đếm
                pass

    def stop(self):
        self.running = False

    def stats(self) -> dict:
        with self._lock:
            s = dict(self._stats)
        s.update(id=self.id, running=self.running, status=self.status, error=self.error,
                 source=str(self.source), kind=self.kind,
                 source_ok=bool(self.fs and self.fs.ok))
        return s

    def frame(self) -> Optional[bytes]:
        with self._lock:
            return self.last_jpg


# --------------------------------------------------------------------------- #
# Health + snapshot
# --------------------------------------------------------------------------- #
@app.get("/healthz")
def healthz():
    return {"status": "ok", "jobs": len(_JOBS), "vectordb": _VDB.backend}


@app.get("/api/snapshot")
def api_snapshot(source: str, w: int = 960):
    """Lấy 1 FRAME từ nguồn (để vẽ vạch/vùng). Trả JPEG."""
    fr = grab_snapshot(source)
    if fr is None:
        raise HTTPException(status_code=502, detail=f"Không lấy được frame từ nguồn: {source!r}")
    import cv2

    if w and fr.shape[1] > w:
        fr = cv2.resize(fr, (w, round(w * fr.shape[0] / fr.shape[1])))
    jpg = encode_jpeg(fr)
    if jpg is None:
        raise HTTPException(status_code=500, detail="Không mã hoá được JPEG")
    return Response(content=jpg, media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@app.post("/api/jobs")
def create_job(req: JobRequest):
    job_id = uuid.uuid4().hex[:8]
    try:
        job = Job(job_id, req)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cấu hình sai: {e}")
    with _LOCK:
        _JOBS[job_id] = job
    job.start()
    return job.stats()


@app.get("/api/jobs")
def list_jobs():
    with _LOCK:
        return [j.stats() for j in _JOBS.values()]


def _get(job_id: str) -> "Job":
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job không tồn tại")
    return job


@app.get("/api/jobs/{job_id}")
def job_stats(job_id: str):
    return _get(job_id).stats()


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    _get(job_id).stop()
    return {"id": job_id, "stopped": True}


@app.get("/api/jobs/{job_id}/frame.jpg")
def job_frame(job_id: str):
    f = _get(job_id).frame()
    if f is None:
        raise HTTPException(status_code=503, detail="chưa có frame (đang nạp model/kết nối)")
    return Response(content=f, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/mjpeg")
def job_mjpeg(job_id: str):
    job = _get(job_id)

    def gen():
        while True:
            f = job.frame()
            if f is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
            if not job.running and job.frame() is None:
                break
            time.sleep(0.04)                     # ~25 fps: đủ mượt cho video phát đúng FPS gốc

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------- #
# Vector DB: sự kiện + tìm kiếm/ReID
# --------------------------------------------------------------------------- #
@app.get("/api/vectordb")
def vectordb_status():
    return _VDB.status()


# --------------------------------------------------------------------------- #
# Mock API cho kết quả đếm (Dựa theo API_SCHEMA.md)
# --------------------------------------------------------------------------- #
@app.get("/api/v1/counting-result")
def get_mock_counting_result():
    import json
    import os
    # Đường dẫn trỏ tới file mock_responses.json (do Docker WORKDIR là /app)
    mock_path = "docs/mock_responses.json"
    if os.path.exists(mock_path):
        with open(mock_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Trả về kịch bản success mặc định
            return data.get("success", {})
    return {"status": "error", "message": "Không tìm thấy file mock_responses.json"}



@app.get("/api/events")
def api_events(source: Optional[str] = None, limit: int = 20):
    return {"recent_events": _VDB.recent(limit, source=source), **_VDB.status()}


@app.post("/api/search/similar")
async def api_search_similar(file: UploadFile = File(...), limit: int = 5):
    """Upload 1 ảnh → tìm vật GIỐNG NHẤT đã ghi (ReID)."""
    import cv2
    import numpy as np

    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Ảnh không hợp lệ")
    return JSONResponse({"results": _VDB.search(embed_crop(img), limit)})


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


# --------------------------------------------------------------------------- #
_INDEX_HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisionOS · Đếm theo camera</title>
<style>
 body{font-family:system-ui,Arial;margin:0;background:#0f172a;color:#e2e8f0}
 header{padding:14px 20px;background:#1e293b;font-size:18px;font-weight:700}
 .wrap{display:flex;flex-wrap:wrap;gap:18px;padding:18px}
 .card{background:#1e293b;border-radius:10px;padding:16px;min-width:340px}
 label{display:block;font-size:13px;margin:8px 0 3px;color:#94a3b8}
 input,select{width:100%;padding:7px;border-radius:6px;border:1px solid #334155;background:#0f172a;color:#e2e8f0}
 button{margin-top:10px;padding:9px 14px;border:0;border-radius:7px;background:#2563eb;color:#fff;font-weight:600;cursor:pointer}
 button.alt{background:#0e7490}button.stop{background:#b91c1c}button.warn{background:#a16207}
 canvas{max-width:100%;border-radius:8px;background:#000;cursor:crosshair;touch-action:none}
 img{max-width:100%;border-radius:8px;background:#000}
 .stat{display:inline-block;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px 12px;margin:4px 6px 0 0}
 .stat b{font-size:20px;color:#38bdf8}
 small{color:#64748b}.hint{color:#fbbf24}
 #events div{font-size:12px;border-bottom:1px solid #334155;padding:3px 0}
</style></head><body>
<header>🎥 VisionOS · AI service đếm người & phương tiện (YOLO + supervision + vector DB)</header>
<div class="wrap">
 <div class="card" style="flex:1">
  <label>Nguồn camera (RTSP / HTTP / đường dẫn file / "0" = webcam)</label>
  <input id="source" value="https://media.roboflow.com/supervision/video-examples/people-walking.mp4" placeholder='rtsp://... hoặc /data/video.mp4 hoặc 0'>
  <button class="alt" onclick="snap()">📷 Lấy frame để vẽ</button>
  <label>Đối tượng</label>
  <select id="prompt"><option value="person">person (người)</option><option value="vehicle">vehicle (mọi xe)</option><option value="car">car</option><option value="truck">truck</option><option value="bus">bus</option></select>
  <label style="display:flex;align-items:center;gap:6px;margin-top:8px;color:#e2e8f0">
   <input type="checkbox" id="grp" style="width:auto"> Gộp mọi loại xe thành “vehicle” (bỏ tick = giữ car/truck/bus mỗi loại 1 màu)
  </label>
  <label>Kiểu đếm — rồi VẼ lên khung bên phải</label>
  <select id="ctype" onchange="resetDraw()">
    <option value="line">Cắt VẠCH (vẽ 2 điểm)</option>
    <option value="zone">Trong VÙNG (vẽ đa giác)</option>
    <option value="fullscreen">Toàn màn hình (không cần vẽ)</option>
  </select>
  <label>max_fps (cao hơn = mượt hơn nếu CPU kịp)</label><input id="fps" value="12">
  <label>Detect mỗi N frame (1 = mượt/đều nhất; tăng = nhanh hơn nhưng có thể giật hơn)</label><input id="dev" value="1">
  <button onclick="startJob()">▶ Bắt đầu đếm</button>
  <button class="stop" onclick="stopJob()">■ Dừng</button>
  <button class="warn" onclick="resetDraw()">↺ Vẽ lại</button>
  <p><small id="msg" class="hint">Nhập nguồn → “Lấy frame” → chọn kiểu đếm → VẼ lên khung → “Bắt đầu đếm”.<br>
   ⚠️ Vạch phải kéo PHỦ HẾT các làn (cả làn ngoài) thì mới đếm đủ.</small></p>
  <hr style="border-color:#334155">
  <label>🔎 Sự kiện gần nhất (vector DB)</label>
  <button class="alt" onclick="loadEvents()">Tải sự kiện</button>
  <div id="events"></div>
 </div>
 <div class="card" style="flex:2">
  <div id="stats"></div>
  <canvas id="cv" width="960" height="540"></canvas>
  <img id="view" style="display:none" alt="">
 </div>
</div>
<script>
let JID=null, poll=null, pts=[], img=new Image(), ctype='line';
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
function msg(t,warn){const m=document.getElementById('msg');m.textContent=t;m.className=warn?'hint':'';}
function snap(){
 const s=document.getElementById('source').value; if(!s){msg('Nhập nguồn trước',1);return;}
 msg('Đang lấy frame…');
 img=new Image();
 img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;resetDraw();msg('Đã có frame — chọn kiểu đếm rồi VẼ.',1);};
 img.onerror=()=>msg('Không lấy được frame (kiểm tra nguồn).',1);
 img.src='/api/snapshot?w=960&source='+encodeURIComponent(s)+'&t='+Date.now();
}
function redraw(){
 if(!img.src)return; ctx.clearRect(0,0,cv.width,cv.height); ctx.drawImage(img,0,0,cv.width,cv.height);
 ctx.lineWidth=3; ctx.strokeStyle='#facc15'; ctx.fillStyle='#facc15';
 const P=pts.map(p=>[p[0]/100*cv.width,p[1]/100*cv.height]);
 if(ctype==='zone'&&P.length){ctx.strokeStyle='#22c55e';ctx.beginPath();ctx.moveTo(P[0][0],P[0][1]);P.slice(1).forEach(q=>ctx.lineTo(q[0],q[1]));if(P.length>2)ctx.closePath();ctx.stroke();}
 if(ctype==='line'&&P.length===2){ctx.beginPath();ctx.moveTo(P[0][0],P[0][1]);ctx.lineTo(P[1][0],P[1][1]);ctx.stroke();}
 P.forEach(q=>{ctx.beginPath();ctx.arc(q[0],q[1],5,0,7);ctx.fill();});
}
cv.addEventListener('click',e=>{
 if(!img.src||ctype==='fullscreen')return;
 const r=cv.getBoundingClientRect();
 const x=(e.clientX-r.left)/r.width*100, y=(e.clientY-r.top)/r.height*100;
 if(ctype==='line'){ if(pts.length>=2)pts=[]; pts.push([x,y]); }
 else pts.push([x,y]);
 redraw();
});
function resetDraw(){ctype=document.getElementById('ctype').value;pts=[];document.getElementById('view').style.display='none';cv.style.display='';redraw();}
function startJob(){
 const s=document.getElementById('source').value; if(!s){msg('Nhập nguồn',1);return;}
 let body={source:s,prompt:document.getElementById('prompt').value,counting_type:ctype,
   model:'yolo',max_fps:parseFloat(document.getElementById('fps').value)||12,
   detect_every:parseInt(document.getElementById('dev').value)||1,
   group_label:document.getElementById('grp').checked};
 if(ctype==='line'){ if(pts.length!==2){msg('Hãy VẼ 2 điểm cho vạch.',1);return;} body.line=[pts[0][0],pts[0][1],pts[1][0],pts[1][1]]; }
 else if(ctype==='zone'){ if(pts.length<3){msg('Vẽ ≥3 điểm cho vùng.',1);return;} body.zone=pts; }
 msg('Đang tạo job…');
 fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  .then(r=>r.json()).then(j=>{
    if(j.detail){msg('Lỗi: '+(typeof j.detail==='string'?j.detail:JSON.stringify(j.detail)),1);return;}
    JID=j.id; msg('Job '+JID+' — '+(j.status||''),1);
    cv.style.display='none'; const v=document.getElementById('view'); v.style.display='';
    v.src='/api/jobs/'+JID+'/mjpeg?t='+Date.now();
    if(poll)clearInterval(poll); poll=setInterval(refresh,1000);
  }).catch(e=>msg('Lỗi: '+e,1));
}
function stopJob(){ if(JID)fetch('/api/jobs/'+JID+'/stop',{method:'POST'}); }
function refresh(){ if(!JID)return;
 fetch('/api/jobs/'+JID).then(r=>r.json()).then(s=>{
  let h=''; const add=(k,v)=>h+='<span class="stat">'+k+'<br><b>'+v+'</b></span>';
  if(s.counting_type==='line'){add((s.in_label||'IN'),s.in||0);add((s.out_label||'OUT'),s.out||0);add('Tổng qua vạch',s.total||0);}
  else if(s.counting_type==='fullscreen'){add('Đang trong khung',s.in_frame||0);add('Đỉnh',s.peak||0);add('Tổng',s.total||0);}
  else{add('Trong vùng',s.in_zone||0);add('Đỉnh vùng',s.zone_peak||0);}
  add('Số vật (track)',s.tracks||0); add('det/frame',s.det_per_frame||0); add('fps',s.fps||0);
  h+='<div><small>'+(s.status||'')+(s.error?(' · LỖI: '+s.error):'')+' · camera '+(s.source_ok?'OK':'chờ')+'</small></div>';
  document.getElementById('stats').innerHTML=h;
 });
}
function loadEvents(){
 let src = document.getElementById('source').value;
 fetch('/api/events?limit=50&source=' + encodeURIComponent(src)).then(r=>r.json()).then(d=>{
  let eventsObj = {};
  (d.recent_events||[]).forEach(e=>{
    const p=e.payload||{};
    let groupKey = Math.floor(p.ts); 
    if(!eventsObj[groupKey]) {
      eventsObj[groupKey] = { ts: p.ts, full_frame: p.full_frame_base64, video_url: p.video_url, items: [] };
    }
    eventsObj[groupKey].items.push(p);
  });
  
  let h='<small>Vector DB: '+d.backend+' · '+d.events+' bản ghi</small><br><br>';
  let groups = Object.values(eventsObj).sort((a,b)=>b.ts - a.ts);
  
  groups.forEach((g, idx) => {
    let timeStr = new Date(g.ts * 1000).toLocaleTimeString();
    h += `<details ${idx === 0 ? 'open' : ''} style="margin-bottom: 10px; background: #222; padding: 10px; border-radius: 6px;">
            <summary style="cursor: pointer; font-weight: bold; outline: none;">⏱ Sự kiện lúc ${timeStr} - Phát hiện ${g.items.length} đối tượng</summary>
            <div style="margin-top: 10px;">`;
    if (g.video_url) {
      h += `<video src="${g.video_url}" controls autoplay muted loop style="width: 100%; border-radius: 4px; margin-bottom: 10px;"></video>`;
    } else if (g.full_frame) {
      h += `<img src="${g.full_frame}" style="width: 100%; border-radius: 4px; margin-bottom: 10px;" />`;
    }
    h += `<div style="display: flex; gap: 10px; overflow-x: auto; padding-bottom: 10px;">`;
    g.items.forEach(p => {
      let img = p.image_base64 ? `<img src="${p.image_base64}" style="width: 60px; height: 60px; object-fit: cover; border-radius: 4px;" />` : '';
      h += `<div style="text-align: center; font-size: 0.8em; background: #333; padding: 5px; border-radius: 4px; min-width: 60px;">
              ${img}
              <div style="margin-top: 4px;">#${p.track_id}</div>
              <div style="color: #aaa;">${p.class_name}</div>
            </div>`;
    });
    h += `</div></div></details>`;
  });
  document.getElementById('events').innerHTML=h;
 });
}
</script></body></html>"""
