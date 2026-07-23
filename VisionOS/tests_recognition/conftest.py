"""Cho phép ``import recognition`` / ``import la_counting`` khi chạy pytest.

Chèn thư mục cha (VisionOS/, nơi chứa các package) vào sys.path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
