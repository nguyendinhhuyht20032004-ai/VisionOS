"""Cho phép ``import la_counting`` và ``import fakes`` khi chạy pytest.

Chèn 2 thư mục vào sys.path:
  * thư mục cha (VisionOS/, chứa package ``la_counting``), và
  * chính thư mục ``tests/`` (chứa ``fakes.py``) — để ``import fakes`` chạy được
    mọi nơi mà KHÔNG phụ thuộc ``tests`` có phải package hay không (trên Kaggle
    Python 3.12, một package ``tests`` khác trong site-packages có thể che mất).
"""

import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # VisionOS/  -> import la_counting
sys.path.insert(0, _HERE)                   # tests/     -> import fakes

# ByteTrack bị deprecate ở supervision >=0.28 nhưng vẫn hoạt động; im lặng để
# output test sạch (notebook gốc cũng dùng sv.ByteTrack).
warnings.filterwarnings("ignore", message=".*ByteTrack.*", category=Warning)
