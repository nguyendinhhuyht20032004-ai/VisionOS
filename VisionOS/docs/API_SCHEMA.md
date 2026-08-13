# 📑 API Schema for VisionOS Counting Service

## 1️⃣ Request schema (used for **/api/jobs**)
```json
{
  "source": "string",                 // URL/RTSP/file path or "0" for webcam
  "prompt": "string",                 // e.g. "person", "vehicle", "car", "truck", "bus"
  "counting_type": "string",          // "line" | "zone" | "fullscreen"
  "line": [x1, y1, x2, y2]?,           // percentages (0‑100) – required when counting_type = "line"
  "zone": [[x, y], …]?,                // list of points – required when counting_type = "zone"
  "model": "string",                  // currently only "yolo"
  "resolution": [width, height],      // optional, default [960, 540]
  "max_fps": number,                   // upper bound for output FPS (default 12)
  "confidence": number,                // YOLO confidence threshold (default 0.25)
  "detect_every": integer,             // run YOLO every N frames (default 1)
  "group_label": boolean,              // merge all vehicle classes into a single label
  "in_label": "string",               // label for “in” direction (default "IN")
  "out_label": "string",              // label for “out” direction (default "OUT")
  "anchor": "string?",                // unused – reserved for future
  "record_events": boolean             // store appearance events in vector DB (default true)
}
```
> All numeric values are **percentage‑based** (0‑100) for `line`/`zone` so they are resolution‑agnostic.

## 2️⃣ Response schema (real‑time job status – **/api/jobs/{id}**)
```json
{
  "id": "string----------------------",
  "running": true|false,
  "status": "string",                 // e.g. "đang chạy", "đã dừng", "lỗi"
  "error": "string?",
  "source": "string",
  "kind": "string",                    // detector type (e.g. "yolo")
  "source_ok": true|false,
  "total_in": number?,                // total objects that crossed the line into the scene
  "total_out": number?,               // total objects that crossed the line out of the scene
  "current_total": number?,           // objects currently present (for fullscreen counting)
  "in": number?,                      // count for the IN side of a line
  "out": number?,                     // count for the OUT side of a line
  "in_zone": number?,                 // count inside a polygon zone
  "zone_peak": number?,               // peak count seen inside a zone
  "tracks": number,                   // number of active tracked objects
  "det_per_frame": number,            // average detections per processed frame
  "fps": number,                      // effective processing frame‑rate
  "timestamp": "ISO‑8601 string"
}
```
All fields marked with `?` are present only for the corresponding `counting_type`.

## 3️⃣ Hard‑coded example response (used for demo endpoint **/api/example-response**)
```json
{
  "status": "success",
  "job_id": "demo_job_001",
  "data": {
    "total_in": 12,
    "total_out": 5,
    "current_total": 7,
    "timestamp": "2026-08-10T14:30:00Z"
  },
  "message": "Demo hard‑coded counting result"
}
```
The above format matches the `CountingResponse` model defined in `app.py`.

---
*This document is intended for developers who need to integrate the VisionOS counting service into external systems. The schemas are expressed in JSON‑Schema‑compatible style for easy validation.*
