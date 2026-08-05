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
from pydantic import BaseModel

from .builder import get_detector, make_scenario
from .camera import FrameSource, encode_jpeg, grab_snapshot
from .engine import StreamingCounter
from .vectordb import VectorStore, embed_crop, make_event_payload

app = FastAPI(title="VisionOS · AI Counting Service")

_JOBS: "Dict[str, Job]" = {}
_LOCK = threading.Lock()
_VDB = VectorStore()          # Qdrant nếu có QDRANT_URL, không thì in-memory


class JobRequest(BaseModel):
    source: str
    prompt: str = "person"
    counting_type: str = "line"                  # "line" | "zone" | "fullscreen"
    line: Optional[List[float]] = None           # [x1,y1,x2,y2] % (người dùng vẽ)
    zone: Optional[List[List[float]]] = None     # [[x,y],…] % (người dùng vẽ)
    model: str = "yolo"                          # người/xe → YOLO
    resolution: List[int] = [960, 540]
    max_fps: float = 8.0
    confidence: float = 0.25
    detect_every: int = 1                         # chạy YOLO mỗi N frame (>1 = nhanh hơn trên CPU)
    group_label: bool = False                     # True = gộp mọi loại xe → 1 nhãn "vehicle";
                                                  # False (mặc định) = GIỮ phân loại car/truck/bus/van/ambulance…
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
        from ..detectors.yolo_nas import COCO_ALIASES
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
            self.counter = StreamingCounter(self.scenario, detector,
                                            resolution=self.scenario.resolution,
                                            detect_every=self.req.detect_every,
                                            merge_label=self.merge_label)
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
        for i, tid in enumerate(det.tracker_id):
            if tid is None:
                continue
            tid = int(tid)
            if tid in self._recorded:
                continue
            x1, y1, x2, y2 = (int(v) for v in det.xyxy[i])
            crop = c.last_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if getattr(crop, "size", 0) == 0:
                continue
            self._recorded.add(tid)
            nm = str(names[i]) if names is not None else self.req.prompt
            try:
                _VDB.add_event(embed_crop(crop),
                               make_event_payload(tid, nm, self.source, self.scenario.counting_type))
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


def _snapshot_hint(source: str) -> str:
    """Gợi ý NGUYÊN NHÂN không lấy được frame (theo loại nguồn) — hay gặp khi chạy Docker."""
    s = str(source).strip()
    if s.isdigit():
        return ("Webcam (số thiết bị) KHÔNG dùng được trong Docker — nhất là trên Mac/Windows. "
                "Hãy chạy service bằng pip (không Docker) để dùng webcam, hoặc dùng RTSP/HTTP.")
    if s.startswith("rtsp://"):
        return ("Đã thử CẢ TCP lẫn UDP mà không có frame. RTSP test MIỄN PHÍ (test.rtsp.stream…) "
                "hay HẾT HẠN / giới hạn kết nối → lấy URL mới. Kiểm tra URL/credential; xác minh "
                "stream còn sống bằng VLC hoặc `ffprobe <url>` trên máy host (ngoài Docker).")
    if s.startswith(("http://", "https://", "rtmp://")):
        return "Kiểm tra URL trỏ THẲNG tới video (mp4/mjpeg) + mạng tới được TỪ container."
    import os

    if not os.path.exists(s):
        return ("File KHÔNG có trong container. Đặt video vào thư mục ./data (cạnh docker-compose.yml) "
                "rồi nhập đường dẫn /data/<tên>.mp4 (đã mount sẵn). Hoặc dùng URL http(s) tới video.")
    return "Đọc được đường dẫn nhưng không giải mã được video (thử file/URL khác)."


@app.get("/api/snapshot")
def api_snapshot(source: str, w: int = 960):
    """Lấy 1 FRAME từ nguồn (để vẽ vạch/vùng). Trả JPEG."""
    fr = grab_snapshot(source)
    if fr is None:
        raise HTTPException(status_code=502,
                            detail=f"Không lấy được frame từ {source!r}. {_snapshot_hint(source)}")
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
            time.sleep(0.06)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------- #
# Vector DB: sự kiện + tìm kiếm/ReID
# --------------------------------------------------------------------------- #
@app.get("/api/vectordb")
def vectordb_status():
    return _VDB.status()


@app.get("/api/events")
def api_events(limit: int = 20):
    return {"events": _VDB.recent(limit), **_VDB.status()}


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
  <input id="source" placeholder='rtsp://... hoặc /data/video.mp4 hoặc 0'>
  <button class="alt" onclick="snap()">📷 Lấy frame để vẽ</button>
  <label>Đối tượng</label>
  <select id="prompt"><option value="person">person (người)</option><option value="vehicle">vehicle (mọi xe)</option><option value="car">car</option><option value="truck">truck</option><option value="bus">bus</option><option value="ambulance">ambulance (cần model OIV7)</option><option value="van">van (cần model OIV7)</option></select>
  <label style="display:flex;align-items:center;gap:6px;margin-top:8px;color:#e2e8f0">
   <input type="checkbox" id="grp" style="width:auto"> Gộp mọi loại xe thành “vehicle” (bỏ tick = giữ car/truck/bus/van/ambulance)
  </label>
  <label>Kiểu đếm — rồi VẼ lên khung bên phải</label>
  <select id="ctype" onchange="resetDraw()">
    <option value="line">Cắt VẠCH (vẽ 2 điểm)</option>
    <option value="zone">Trong VÙNG (vẽ đa giác)</option>
    <option value="fullscreen">Toàn màn hình (không cần vẽ)</option>
  </select>
  <label>max_fps</label><input id="fps" value="8">
  <label>Detect mỗi N frame (CPU chậm → để 2–3 cho mượt hơn)</label><input id="dev" value="1">
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
 fetch('/api/snapshot?w=960&source='+encodeURIComponent(s)+'&t='+Date.now())
  .then(async r=>{ if(!r.ok){let d='';try{d=(await r.json()).detail;}catch(e){} throw new Error(d||('HTTP '+r.status));} return r.blob(); })
  .then(b=>{ img=new Image();
    img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;resetDraw();msg('Đã có frame — chọn kiểu đếm rồi VẼ.',1);};
    img.src=URL.createObjectURL(b); })
  .catch(e=>msg('Không lấy được frame — '+e.message,1));
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
   model:'yolo',max_fps:parseFloat(document.getElementById('fps').value)||8,
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
 fetch('/api/events?limit=20').then(r=>r.json()).then(d=>{
  let h='<small>Vector DB: '+d.backend+' · '+d.events+' sự kiện</small>';
  (d.events||[]).forEach(e=>{const p=e.payload||{};h+='<div>#'+p.track_id+' · '+p.class_name+' · '+(p.counting_type||'')+'</div>';});
  document.getElementById('events').innerHTML=h;
 });
}
</script></body></html>"""
