"""Ứng dụng FastAPI cho service đếm theo camera.

Endpoint:
  POST /api/jobs            — tạo job đếm trên 1 nguồn camera → trả job.
  GET  /api/jobs            — liệt kê job.
  GET  /api/jobs/{id}       — số đếm hiện tại (JSON).
  GET  /api/jobs/{id}/frame.jpg — frame annotate mới nhất (JPEG).
  GET  /api/jobs/{id}/mjpeg — luồng MJPEG annotate (dán vào <img>).
  POST /api/jobs/{id}/stop  — dừng job.
  GET  /                    — trang web điều khiển + xem trực tiếp.

Chạy: ``python run_service.py --host 0.0.0.0 --port 8000``.
fastapi/uvicorn import ở đây → chỉ cần khi CHẠY service (``pip install fastapi uvicorn``).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from .builder import get_detector, make_scenario
from .camera import FrameSource
from .engine import StreamingCounter

app = FastAPI(title="VisionOS · Camera Counting Service")

_JOBS: "Dict[str, Job]" = {}
_LOCK = threading.Lock()


class JobRequest(BaseModel):
    source: str                                  # rtsp://… | http://… | /path.mp4 | "0" (webcam)
    prompt: str = "person"                       # đối tượng (YOLO) hoặc mô tả (LocateAnything)
    counting_type: str = "line"                  # "line" | "zone"
    line: Optional[List[float]] = None           # [x1,y1,x2,y2] % (0..100)
    zone: Optional[List[List[float]]] = None     # [[x,y],…] % (≥3 đỉnh)
    model: str = "auto"                          # "yolo" | "locate" | "auto"
    resolution: List[int] = [960, 540]
    max_fps: float = 4.0                          # giới hạn tốc độ xử lý (LA chậm → để thấp)
    confidence: float = 0.25
    in_label: str = "IN"
    out_label: str = "OUT"
    anchor: Optional[str] = None


class Job:
    def __init__(self, job_id: str, req: JobRequest):
        self.id = job_id
        self.req = req
        self.scenario, self.kind = make_scenario(
            req.prompt, req.counting_type, req.line, req.zone,
            tuple(req.resolution), req.in_label, req.out_label, req.anchor, req.model)
        self.source = req.source
        self.fs: Optional[FrameSource] = None
        self.counter: Optional[StreamingCounter] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.status = "khởi tạo"
        self.error: Optional[str] = None
        self.last_jpg: Optional[bytes] = None
        self._stats: dict = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- #
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        import cv2

        try:
            self.status = "đang nạp model (" + self.kind + ")"
            detector = get_detector(self.kind, self.req.confidence)
            self.counter = StreamingCounter(self.scenario, detector,
                                            resolution=self.scenario.resolution)
            self.status = "đang kết nối camera"
            self.fs = FrameSource(self.source).start()
            self.status = "đang chạy"
            interval = 1.0 / self.req.max_fps if self.req.max_fps > 0 else 0.0
            last = 0.0
            while self.running:
                fr = self.fs.read()
                if fr is None:
                    if self.fs.error and not self.fs.alive:
                        self.error = self.fs.error
                        break
                    time.sleep(0.05)
                    continue
                now = time.time()
                if interval and now - last < interval:
                    time.sleep(min(0.02, interval - (now - last)))
                    continue
                last = now
                out = self.counter.process(fr)
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self.last_jpg = buf.tobytes()
                        self._stats = self.counter.stats()
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.running = False
            self.status = "lỗi" if self.error else "đã dừng"
            if self.fs is not None:
                self.fs.stop()

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
# API
# --------------------------------------------------------------------------- #
@app.post("/api/jobs")
def create_job(req: JobRequest):
    job_id = uuid.uuid4().hex[:8]
    try:
        job = Job(job_id, req)
    except Exception as e:  # noqa: BLE001 — cấu hình sai (vd vùng <3 đỉnh)
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
    job = _get(job_id)
    job.stop()
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
            time.sleep(0.06)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


# --------------------------------------------------------------------------- #
_INDEX_HTML = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VisionOS · Đếm theo camera</title>
<style>
 body{font-family:system-ui,Arial;margin:0;background:#0f172a;color:#e2e8f0}
 header{padding:14px 20px;background:#1e293b;font-size:18px;font-weight:700}
 .wrap{display:flex;flex-wrap:wrap;gap:18px;padding:18px}
 .card{background:#1e293b;border-radius:10px;padding:16px;min-width:320px}
 label{display:block;font-size:13px;margin:8px 0 3px;color:#94a3b8}
 input,select{width:100%;padding:7px;border-radius:6px;border:1px solid #334155;background:#0f172a;color:#e2e8f0}
 button{margin-top:12px;padding:9px 16px;border:0;border-radius:7px;background:#2563eb;color:#fff;font-weight:600;cursor:pointer}
 button.stop{background:#b91c1c}
 img{max-width:100%;border-radius:8px;background:#000}
 .stat{display:inline-block;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px 12px;margin:4px 6px 0 0}
 .stat b{font-size:20px;color:#38bdf8}
 small{color:#64748b}
</style></head><body>
<header>🎥 VisionOS · Service đếm theo camera (LocateAnything-3B + YOLO)</header>
<div class="wrap">
 <div class="card" style="flex:1">
  <label>Nguồn camera (RTSP / HTTP / đường dẫn file / "0" = webcam)</label>
  <input id="source" placeholder="rtsp://... hoặc /kaggle/working/video.mp4 hoặc 0">
  <label>Đối tượng cần đếm (prompt)</label>
  <input id="prompt" value="person" placeholder="person / car / object / carton box / tomato">
  <label>Kiểu đếm</label>
  <select id="ctype"><option value="line">Cắt vạch (vào/ra)</option><option value="zone">Trong vùng</option></select>
  <label>Vạch % [x1,y1,x2,y2] (kiểu 'line')</label>
  <input id="line" value="0,50,100,50">
  <label>Vùng % [[x,y],...] (kiểu 'zone')</label>
  <input id="zone" value="[[20,20],[80,20],[80,80],[20,80]]">
  <label>Model</label>
  <select id="model"><option value="auto">auto (tự chọn)</option><option value="yolo">YOLO (nhanh: người/xe)</option><option value="locate">LocateAnything (open-vocab: sản phẩm)</option></select>
  <label>max_fps (LA chậm → 2-4)</label>
  <input id="fps" value="4">
  <button onclick="startJob()">▶ Bắt đầu</button>
  <button class="stop" onclick="stopJob()">■ Dừng</button>
  <p><small id="msg"></small></p>
 </div>
 <div class="card" style="flex:2">
  <div id="stats"></div>
  <img id="view" src="" alt="(chưa chạy)">
 </div>
</div>
<script>
let JID=null, poll=null;
function startJob(){
 let body={source:document.getElementById('source').value,
   prompt:document.getElementById('prompt').value,
   counting_type:document.getElementById('ctype').value,
   model:document.getElementById('model').value,
   max_fps:parseFloat(document.getElementById('fps').value)||4};
 try{body.line=document.getElementById('line').value.split(',').map(Number);}catch(e){}
 try{body.zone=JSON.parse(document.getElementById('zone').value);}catch(e){}
 fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  .then(r=>r.json()).then(j=>{
    if(j.detail){document.getElementById('msg').textContent='Lỗi: '+j.detail;return;}
    JID=j.id; document.getElementById('msg').textContent='Job '+JID+' — '+(j.status||'');
    document.getElementById('view').src='/api/jobs/'+JID+'/mjpeg?t='+Date.now();
    if(poll)clearInterval(poll); poll=setInterval(refresh,1000);
  }).catch(e=>document.getElementById('msg').textContent='Lỗi: '+e);
}
function stopJob(){ if(!JID)return; fetch('/api/jobs/'+JID+'/stop',{method:'POST'}); }
function refresh(){ if(!JID)return;
 fetch('/api/jobs/'+JID).then(r=>r.json()).then(s=>{
  let h='';
  const add=(k,v)=>h+='<span class="stat">'+k+'<br><b>'+v+'</b></span>';
  if(s.counting_type==='line'){add((s.in_label||'IN'),s.in||0);add((s.out_label||'OUT'),s.out||0);add('Tổng qua vạch',s.total||0);}
  else{add('Trong vùng',s.in_zone||0);add('Đỉnh vùng',s.zone_peak||0);}
  add('Số vật (track)',s.tracks||0); add('det/frame',s.det_per_frame||0); add('fps',s.fps||0);
  h+='<div><small>Trạng thái: '+(s.status||'')+(s.error?(' · LỖI: '+s.error):'')+' · camera '+(s.source_ok?'OK':'chờ')+'</small></div>';
  document.getElementById('stats').innerHTML=h;
 });
}
</script></body></html>"""
