#!/usr/bin/env bash
# Tải các VIDEO TEST công khai (Roboflow supervision) về ./data để dùng trong service.
# Đây đúng là các video đã dùng khi test mô hình (người ở sảnh/ga + xe/giao thông) —
# link CÔNG KHAI, ỔN ĐỊNH (khác RTSP test miễn phí hay hết hạn).
#
# Chạy TRÊN MÁY BẠN (Mac/Linux), tại thư mục repo:
#     bash scripts/fetch_test_videos.sh            # tải vào ./data
#     bash scripts/fetch_test_videos.sh /duong/dan # tải vào thư mục khác
#
# Sau đó trong web (http://localhost:8000) nhập nguồn: /data/<tên>.mp4
set -euo pipefail

DEST="${1:-data}"
mkdir -p "$DEST"
BASE="https://media.roboflow.com/supervision/video-examples"

# Tên file : mô tả (bài toán)
VIDEOS=(
  "vehicles.mp4|Xe cao tốc (top-down) — đếm xe"
  "vehicles-2.mp4|Xe giao lộ nhiều làn — đếm xe/loại xe"
  "people-walking.mp4|Người đi bộ lối đi — đếm người vào/ra"
  "subway.mp4|Người ở SẢNH ga tàu — đếm người (bài chính)"
  "grocery-store.mp4|Người trong siêu thị — đếm người/vùng"
  "market-square.mp4|Người ở quảng trường — đếm người/vùng"
)

echo "⬇️  Tải video test vào: $DEST/"
for row in "${VIDEOS[@]}"; do
  f="${row%%|*}"; desc="${row##*|}"
  if [ -s "$DEST/$f" ]; then
    echo "  ✓ đã có  $DEST/$f  ($desc)"
    continue
  fi
  echo "  ⬇️  $f  — $desc"
  if curl -fL --retry 3 --retry-delay 2 -o "$DEST/$f" "$BASE/$f"; then
    echo "     ✅ $DEST/$f"
  else
    echo "     ❌ lỗi tải $f (kiểm tra mạng, hoặc tải tay từ: $BASE/$f)"
  fi
done

echo ""
echo "Xong. Trong web (http://localhost:8000) nhập nguồn dạng:"
echo "   /data/subway.mp4        (đếm người ở sảnh ga)"
echo "   /data/vehicles.mp4      (đếm xe cao tốc)"
echo "   /data/vehicles-2.mp4    (đếm xe giao lộ — thử prompt 'truck'/'bus'/'xe cấp cứu')"
