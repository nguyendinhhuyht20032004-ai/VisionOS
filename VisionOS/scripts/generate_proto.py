#!/usr/bin/env python3
"""Generate Python gRPC stubs từ overlay.proto.

Chạy:
    python scripts/generate_proto.py

Output: recognition/service/proto/overlay_pb2.py + overlay_pb2_grpc.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proto_dir = os.path.join(root, "proto")
    proto_file = os.path.join(proto_dir, "overlay.proto")
    out_dir = os.path.join(root, "recognition", "service", "proto")

    if not os.path.isfile(proto_file):
        print(f"Khong tim thay {proto_file}")
        return 1

    os.makedirs(out_dir, exist_ok=True)

    # Tạo __init__.py nếu chưa có
    init_file = os.path.join(out_dir, "__init__.py")
    if not os.path.isfile(init_file):
        with open(init_file, "w") as f:
            f.write("# Generated gRPC stubs for overlay.proto\n")

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        proto_file,
    ]

    print(f"Generating gRPC stubs...")
    print(f"   Proto:  {proto_file}")
    print(f"   Output: {out_dir}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"protoc failed:\n{result.stderr}")
        return 1

    # Fix import path trong generated file (protoc sinh import 'overlay_pb2'
    # nhưng khi nằm trong package cần 'from . import overlay_pb2')
    grpc_file = os.path.join(out_dir, "overlay_pb2_grpc.py")
    if os.path.isfile(grpc_file):
        with open(grpc_file, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace(
            "import overlay_pb2 as overlay__pb2",
            "from . import overlay_pb2 as overlay__pb2",
        )
        with open(grpc_file, "w", encoding="utf-8") as f:
            f.write(content)

    print("Done! Generated files:")
    for fn in os.listdir(out_dir):
        print(f"   {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
