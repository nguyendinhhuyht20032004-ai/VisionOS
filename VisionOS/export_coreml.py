#!/usr/bin/env python3
"""Export YOLOv8 sang CoreML (.mlpackage) cho Apple Silicon.

Chạy trên macOS hoặc Linux (cần coremltools). Model CoreML chạy trên
Apple Neural Engine (ANE) nhanh hơn 3-5x so với PyTorch CPU thuần.

Dùng:
    python export_coreml.py                           # FP16 (mặc định, tối ưu ANE)
    python export_coreml.py --weights yolov8m.pt      # chọn model
    python export_coreml.py --int8                    # INT8 (nhỏ hơn, hơi kém chính xác)
    python export_coreml.py --nms                     # gộp NMS vào model (nhanh hơn, cố định tham số)

Sau khi export xong, đặt env YOLO_WEIGHTS trỏ tới file .mlpackage:
    export YOLO_WEIGHTS=yolov8m.mlpackage
    python run_service.py
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export YOLOv8 sang CoreML cho Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--weights", default="yolov8m.pt",
                    help="File trọng số PyTorch (default: yolov8m.pt)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Kích thước ảnh đầu vào (default: 640)")
    ap.add_argument("--half", action="store_true", default=True,
                    help="Export FP16 — tối ưu cho ANE (default: True)")
    ap.add_argument("--no-half", dest="half", action="store_false",
                    help="Export FP32 (chậm hơn, chính xác hơn một chút)")
    ap.add_argument("--nms", action="store_true",
                    help="Gộp NMS vào model (nhanh hơn nhưng không đổi được NMS params lúc chạy)")
    ap.add_argument("--int8", action="store_true",
                    help="Lượng tử hóa INT8 (file nhỏ hơn ~2x, hơi giảm mAP)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Cần cài ultralytics:  pip install ultralytics")
        return 1

    print(f"Loading {args.weights} ...")
    model = YOLO(args.weights)

    export_kw = dict(format="coreml", imgsz=args.imgsz, half=args.half, nms=args.nms)
    if args.int8:
        export_kw["int8"] = True
        export_kw["half"] = False

    precision = "INT8" if args.int8 else ("FP16" if args.half else "FP32")
    print(f"Exporting to CoreML ({precision}, imgsz={args.imgsz}, nms={args.nms}) ...")

    path = model.export(**export_kw)

    print(f"\nExport thanh cong: {path}")
    print(f"\nCach dung:")
    print(f"  export YOLO_WEIGHTS={path}")
    print(f"  python run_service.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
