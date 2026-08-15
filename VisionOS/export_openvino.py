#!/usr/bin/env python3
"""Export YOLOv8 sang OpenVINO IR cho tăng tốc inference trên CPU Intel.

OpenVINO (Open Visual Inference and Neural Network Optimization) nén + tối ưu
model để chạy nhanh hơn PyTorch thuần trên CPU Intel (và một số GPU Intel).

Dùng:
    python export_openvino.py                           # FP16 (mặc định)
    python export_openvino.py --weights yolov8m.pt      # chọn model
    python export_openvino.py --int8                    # INT8 (nhỏ hơn, nhanh hơn, hơi kém chính xác)

Sau khi export xong, đặt env YOLO_WEIGHTS trỏ tới thư mục model:
    export YOLO_WEIGHTS=yolov8m_openvino_model
    python run_service.py

Lưu ý: Cần cài openvino trước:
    pip install openvino
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export YOLOv8 sang OpenVINO IR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--weights", default="yolov8m.pt",
                    help="File trong so PyTorch (default: yolov8m.pt)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Kich thuoc anh dau vao (default: 640)")
    ap.add_argument("--half", action="store_true", default=True,
                    help="Export FP16 (default: True)")
    ap.add_argument("--no-half", dest="half", action="store_false",
                    help="Export FP32")
    ap.add_argument("--int8", action="store_true",
                    help="Luong tu hoa INT8 (nho + nhanh hon, hoi giam mAP)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Can cai ultralytics:  pip install ultralytics")
        return 1

    try:
        import openvino  # noqa: F401
    except ImportError:
        print("Can cai openvino:  pip install openvino")
        return 1

    print(f"Loading {args.weights} ...")
    model = YOLO(args.weights)

    export_kw = dict(format="openvino", imgsz=args.imgsz, half=args.half)
    if args.int8:
        export_kw["int8"] = True
        export_kw["half"] = False

    precision = "INT8" if args.int8 else ("FP16" if args.half else "FP32")
    print(f"Exporting to OpenVINO IR ({precision}, imgsz={args.imgsz}) ...")

    path = model.export(**export_kw)

    print(f"\nExport thanh cong: {path}")
    print(f"\nCach dung:")
    print(f"  export YOLO_WEIGHTS={path}")
    print(f"  python run_service.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
