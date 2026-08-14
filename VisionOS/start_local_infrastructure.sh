#!/bin/bash
# =========================================================================
# start_local_infrastructure.sh — Bật Redis và MediaMTX NATIVE 100%
# =========================================================================

echo "🚀 Bắt đầu khởi động Cơ sở hạ tầng Native (Không cần Docker)..."

# --- 1. Khởi động Redis Native ---
REDIS_BIN="./local_bin/redis-stable/src/redis-server"
if [ ! -f "$REDIS_BIN" ]; then
    echo "❌ Không tìm thấy Redis Native. Vui lòng build lại!"
    exit 1
fi

echo "🟢 Khởi động Redis ở cổng 6379..."
# Tắt redis cũ nếu đang chạy
pkill -f "redis-server" 2>/dev/null || true
sleep 2 # Đợi quá trình shutdown hoàn tất để giải phóng cổng 6379
nohup "$REDIS_BIN" > local_bin/redis.log 2>&1 &
sleep 1
echo "✅ Redis đã chạy ngầm!"

# --- 2. Khởi động MediaMTX Native ---
MTX_BIN="./local_bin/mediamtx/mediamtx"
if [ ! -f "$MTX_BIN" ]; then
    echo "❌ Không tìm thấy MediaMTX. Vui lòng tải lại!"
    exit 1
fi

echo "🟢 Khởi động MediaMTX ở cổng 8554 (RTSP)..."
pkill -f "mediamtx" 2>/dev/null || true
sleep 2 # Đợi quá trình shutdown hoàn tất để giải phóng cổng 8554
cd local_bin/mediamtx
# Ghi đè cấu hình để mở port API 9997 và RTSP 8554
sed -i '' 's/api: no/api: yes/' mediamtx.yml || true
nohup ./mediamtx > mediamtx.log 2>&1 &
cd ../..
sleep 1
echo "✅ MediaMTX đã chạy ngầm!"

echo ""
echo "🎉 HỆ THỐNG NATIVE ĐÃ SẴN SÀNG!"
echo "Bây giờ bạn có thể bật luồng Video và AI bình thường:"
echo "1. Phát Video: python3 test_people.py"
echo "2. Chạy AI:    ./run_native.sh"
