"""Công cụ VẼ vạch/vùng bằng CHUỘT ngay trong notebook (HTML+JS — chạy trên Kaggle).

Không cần widget backend hay cài thêm gì: render 1 ``<canvas>`` kèm frame thật của
video (nhúng base64) + JavaScript bắt click. Bạn bấm chuột để vẽ **NHIỀU vùng**
(đa giác) và/hoặc **vạch**; công cụ hiện toạ độ theo **% (0–100)** và xuất sẵn chuỗi
``--zone`` / ``--line`` để copy — dán lại cho tôi (đưa vào catalog) hoặc chạy luôn CLI.

Thiết kế KHÔNG có "chế độ": bạn cứ bấm điểm, rồi bấm **Xong VÙNG** (≥3 điểm) hay
**Xong VẠCH** (2 điểm). Với vùng còn có: bấm lại **điểm đầu** (vòng đỏ) hoặc **bấm đúp**
để đóng. Nhờ vậy không còn cảnh "kẹt chế độ vạch nên đóng vùng báo lỗi".

Dùng trong notebook:

    from IPython.display import HTML
    from recognition.draw_tool import draw_for
    HTML(draw_for("subway"))         # tải video 'subway', hiện canvas để vẽ
"""

from __future__ import annotations

import base64
from typing import Optional, Tuple

__all__ = ["frame_data_uri", "draw_html", "draw_for", "save_draw_html"]

_UID = 0


def _next_uid() -> str:
    global _UID
    _UID += 1
    return f"dz{_UID}"


def frame_data_uri(video_path: str, target_index: int = 50,
                   resolution: Optional[Tuple[int, int]] = None) -> Tuple[str, int, int]:
    """Lấy 1 frame CÓ vật (đọc tuần tự tới ``target_index``) → (data-URI JPEG, w, h).

    ``resolution`` (nếu có) resize frame về đúng độ phân giải scenario dùng khi đếm,
    để toạ độ % bạn vẽ khớp CHÍNH XÁC với lúc chạy pipeline.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fr = None
    for _ in range(max(target_index, 0) + 1):
        ok, f = cap.read()
        if not ok:
            break
        fr = f
    cap.release()
    if fr is None:
        raise RuntimeError(f"Không đọc được frame từ: {video_path}")
    if resolution is not None:
        fr = cv2.resize(fr, tuple(resolution))
    h, w = fr.shape[:2]
    ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Mã hoá JPEG frame thất bại")
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    return uri, int(w), int(h)


# --------------------------------------------------------------------------- #
# HTML + JS: canvas vẽ tay, KHÔNG có chế độ. Toàn bộ tính % chạy trong TRÌNH DUYỆT
# (không gọi lại kernel) nên chạy tốt trên Kaggle. {uid} tách biệt nhiều cell/canvas.
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""
<div id="wrap-{uid}" style="font-family:system-ui,Arial,sans-serif;max-width:{dispw}px">
  <div style="margin:6px 0;line-height:2">
    <button id="fzone-{uid}" style="background:#16a34a;color:#fff;font-weight:bold;padding:4px 8px">⬠ Xong VÙNG (≥3 điểm)</button>
    <button id="fline-{uid}" style="background:#3b82f6;color:#fff;font-weight:bold;padding:4px 8px">➖ Xong VẠCH (2 điểm)</button>
    &nbsp;|&nbsp;
    <button id="undo-{uid}">↶ Hoàn tác</button>
    <button id="clear-{uid}">🗑 Xoá hết</button>
    <button id="copy-{uid}">📋 Copy</button>
  </div>
  <div id="hint-{uid}" style="margin:4px 0;padding:4px 8px;border-radius:4px;
       background:#123;color:#0f0;font-weight:bold;display:inline-block"></div>
  <div style="font-size:13px;color:#555;margin:4px 0">
    Bấm chuột lên ảnh để thêm điểm (KHÔNG cần chọn chế độ). &nbsp;<b>VẠCH</b>: 2 điểm →
    nút <b>Xong VẠCH</b>. &nbsp;<b>VÙNG</b>: ≥3 điểm → bấm lại <b>điểm đầu</b> (vòng đỏ) /
    <b>bấm đúp</b> / nút <b>Xong VÙNG</b>. Đóng xong vẽ hình tiếp theo được ngay.
  </div>
  <canvas id="cv-{uid}" style="border:1px solid #888;cursor:crosshair;touch-action:none"></canvas>
  <div style="margin-top:6px">
    <div style="font-size:13px;color:#555">Toạ độ % (bấm 📋 Copy rồi dán cho tôi / chạy CLI):</div>
    <textarea id="out-{uid}" rows="9" readonly
      style="width:100%;font-family:monospace;font-size:12.5px;background:#0d1117;color:#7fdb7f;
             border:1px solid #444;padding:6px;box-sizing:border-box"></textarea>
  </div>
</div>
<script>
(function(){{
  const W={w}, H={h}, DISPW={dispw};
  const img=new Image();
  const canvas=document.getElementById("cv-{uid}");
  const out=document.getElementById("out-{uid}");
  const hint=document.getElementById("hint-{uid}");
  const scale=Math.min(1, DISPW/W);
  canvas.width=W; canvas.height=H;
  canvas.style.width=(W*scale)+"px"; canvas.style.height=(H*scale)+"px";
  const ctx=canvas.getContext("2d");
  let shapes=[], cur=[];                       // KHÔNG có biến 'mode'

  function pct(p){{ return [ +(p[0]/W*100).toFixed(1), +(p[1]/H*100).toFixed(1) ]; }}
  function dist(a,b){{ return Math.hypot(a[0]-b[0], a[1]-b[1]); }}
  function finishZone(){{ if(cur.length>=3){{ shapes.push({{type:"zone",pts:cur.slice()}}); cur=[]; redraw(); return true; }} return false; }}
  function finishLine(){{ if(cur.length>=2){{ shapes.push({{type:"line",pts:[cur[0],cur[cur.length-1]]}}); cur=[]; redraw(); return true; }} return false; }}

  function poly(pts,color,close){{
    ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=3;
    ctx.beginPath();
    pts.forEach((p,i)=> i? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1]));
    if(close && pts.length>2) ctx.closePath();
    ctx.stroke();
    pts.forEach(p=>{{ ctx.beginPath(); ctx.arc(p[0],p[1],5,0,7); ctx.fill(); }});
  }}
  function redraw(){{
    ctx.clearRect(0,0,W,H);
    if(img.complete && img.naturalWidth) ctx.drawImage(img,0,0,W,H);
    ctx.strokeStyle="rgba(255,255,255,0.18)"; ctx.lineWidth=1;
    for(let p=10;p<100;p+=10){{
      ctx.beginPath(); ctx.moveTo(W*p/100,0); ctx.lineTo(W*p/100,H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,H*p/100); ctx.lineTo(W,H*p/100); ctx.stroke();
    }}
    ctx.fillStyle="#ffd000"; ctx.font="12px monospace";
    for(let p=0;p<=100;p+=20){{ ctx.fillText(p, Math.min(W*p/100+2,W-22), 12);
      ctx.fillText(p, 2, Math.min(Math.max(H*p/100+4,12),H-3)); }}
    shapes.forEach(s=> poly(s.pts, s.type==="line"?"#00e0e0":"#00ff66", s.type==="zone"));
    if(cur.length) poly(cur, "#ffd000", false);
    if(cur.length>=3){{                          // vòng đỏ ở điểm đầu = "bấm đây để đóng vùng"
      ctx.strokeStyle="#ff3b3b"; ctx.lineWidth=2;
      ctx.beginPath(); ctx.arc(cur[0][0], cur[0][1], 11, 0, 7); ctx.stroke();
    }}
    updateHint(); updateOut();
  }}
  function updateHint(){{
    let tip = "Đã bấm " + cur.length + " điểm  ·  " + shapes.length + " hình đã chốt";
    if(cur.length>=3) tip += "  →  bấm ĐIỂM ĐẦU (đỏ) / bấm đúp / nút Xong VÙNG";
    else if(cur.length===2) tip += "  →  bấm nút Xong VẠCH (hoặc thêm điểm cho VÙNG)";
    else tip += "  →  bấm ≥3 điểm cho VÙNG, hoặc 2 điểm cho VẠCH";
    hint.textContent = tip;
  }}
  function updateOut(){{
    let nL=0,nZ=0; const rows=["# ==== VẠCH / VÙNG bạn vẽ (toạ độ %) ===="];
    shapes.forEach(s=>{{
      const P=s.pts.map(pct);
      if(s.type==="line"){{ nL++;
        rows.push("L"+nL+"  --line \""+P[0].join(",")+","+P[1].join(",")+"\""); }}
      else {{ nZ++;
        rows.push("Z"+nZ+"  --zone \""+P.map(p=>p.join(",")).join(";")+"\""); }}
    }});
    if(cur.length){{ rows.push("(đang vẽ, "+cur.length+" điểm: "
        +cur.map(pct).map(p=>p.join(",")).join(" ; ")+")"); }}
    const zs=shapes.filter(s=>s.type==="zone");
    if(zs.length){{
      rows.push(""); rows.push("# Dán cho Claude (zone_points_pct):");
      zs.forEach((s,i)=> rows.push("Z"+(i+1)+" = ("
        + s.pts.map(pct).map(p=>"("+p[0]+","+p[1]+")").join(",") + ")"));
    }}
    out.value=rows.join("\n");
  }}

  canvas.addEventListener("click", ev=>{{
    const r=canvas.getBoundingClientRect();
    const x=(ev.clientX-r.left)/r.width*W, y=(ev.clientY-r.top)/r.height*H;
    // Bấm GẦN điểm đầu (khi ≥3 điểm) = ĐÓNG VÙNG (cử chỉ tự nhiên, ngưỡng ~16px hiển thị).
    if(cur.length>=3 && dist([x,y], cur[0]) < 16*(W/r.width)){{ finishZone(); return; }}
    cur.push([Math.round(x),Math.round(y)]);
    redraw();
  }});
  canvas.addEventListener("dblclick", ()=>{{ if(cur.length>=4) cur.pop(); finishZone(); }});

  document.getElementById("fzone-{uid}").onclick=()=>{{ if(!finishZone()) alert("VÙNG cần ≥3 điểm — bạn mới bấm "+cur.length+" điểm. Bấm thêm rồi Xong VÙNG."); }};
  document.getElementById("fline-{uid}").onclick=()=>{{ if(!finishLine()) alert("VẠCH cần 2 điểm — bạn mới bấm "+cur.length+" điểm."); }};
  document.getElementById("undo-{uid}").onclick=()=>{{ if(cur.length) cur.pop(); else if(shapes.length) shapes.pop(); redraw(); }};
  document.getElementById("clear-{uid}").onclick=()=>{{ shapes=[]; cur=[]; redraw(); }};
  document.getElementById("copy-{uid}").onclick=()=>{{
    out.select();
    try{{ document.execCommand("copy"); }}catch(e){{}}
    try{{ navigator.clipboard && navigator.clipboard.writeText(out.value); }}catch(e){{}}
  }};

  img.onload=redraw; img.onerror=redraw; img.src="{uri}";
  redraw();
}})();
</script>
"""


def draw_html(video_path: str, title: str = "", max_width: int = 900,
              target_index: int = 50, resolution: Optional[Tuple[int, int]] = None) -> str:
    """Trả về chuỗi HTML (canvas vẽ tay) cho 1 video path — bọc bằng ``IPython.display.HTML``."""
    uri, w, h = frame_data_uri(video_path, target_index, resolution)
    dispw = min(max_width, w)
    header = f'<h4 style="margin:4px 0">🖊️ Vẽ vạch/vùng — {title}</h4>' if title else ""
    return header + _TEMPLATE.format(uid=_next_uid(), uri=uri, w=w, h=h, dispw=dispw)


def _find(name_or_key: str):
    from .video_catalog import CATALOG

    kw = name_or_key.lower()
    for v in CATALOG:
        hay = (v.name + v.scenario.key + v.filename).lower()
        if kw in hay:
            return v
    raise KeyError(
        f"Không thấy video khớp {name_or_key!r}. "
        f"Khoá hợp lệ: {', '.join(sorted(v.scenario.key for v in CATALOG))}"
    )


def draw_for(name_or_key: str, max_width: int = 900, target_index: int = 50) -> str:
    """Tải video theo tên/khoá trong catalog rồi trả HTML để vẽ (dùng trong notebook)."""
    from .video_catalog import download_video

    v = _find(name_or_key)
    path = download_video(v)
    return draw_html(path, title=f"[{v.task}] {v.name}",
                     max_width=max_width, target_index=target_index,
                     resolution=v.scenario.resolution)


def save_draw_html(name_or_key: str, out_path: str, max_width: int = 900,
                   target_index: int = 50) -> str:
    """Lưu công cụ vẽ ra file .html độc lập (mở bằng trình duyệt hoặc tải từ Kaggle)."""
    html = ("<!doctype html><meta charset='utf-8'><title>Vẽ vạch/vùng</title>"
            + draw_for(name_or_key, max_width=max_width, target_index=target_index))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
