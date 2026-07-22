"""Cho phép ``import la_counting`` khi chạy pytest từ bất kỳ thư mục nào.

Chèn thư mục cha (VisionOS/, nơi chứa package ``la_counting``) vào sys.path.
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ByteTrack bị deprecate ở supervision >=0.28 nhưng vẫn hoạt động; im lặng để
# output test sạch (notebook gốc cũng dùng sv.ByteTrack).
warnings.filterwarnings("ignore", message=".*ByteTrack.*", category=Warning)
