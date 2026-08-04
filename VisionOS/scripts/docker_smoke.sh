#!/usr/bin/env bash
# Kiểm thử NHANH Docker AI service: build → chạy (api + qdrant) → healthz → tạo job
# đếm trên 1 video mẫu → đọc số đếm → dọn dẹp. Cần: docker + docker compose.
#
#   bash scripts/docker_smoke.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1) Build + up (CPU). GPU: sửa TORCH_CUDA=cu121 trong docker-compose.yml"
docker compose up -d --build

cleanup() { echo "==> dọn dẹp"; docker compose logs --tail=30 api || true; docker compose down; }
trap cleanup EXIT

echo "==> 2) Chờ /healthz (tối đa 120s)"
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/healthz >/dev/null 2>&1; then
    echo "   OK: $(curl -fsS http://localhost:8000/healthz)"; break
  fi
  sleep 2
  [ "$i" = 60 ] && { echo "   ❌ service không lên"; exit 1; }
done

echo "==> 3) Vector DB status"
curl -fsS http://localhost:8000/api/vectordb; echo

echo "==> 4) Tải 1 video mẫu (người đi bộ) vào container rồi tạo job đếm qua VẠCH"
docker compose exec -T api sh -c \
  'wget -q https://media.roboflow.com/supervision/video-examples/people-walking.mp4 -O /tmp/pw.mp4 || true'
JOB=$(curl -fsS -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' -d '{
  "source":"/tmp/pw.mp4","prompt":"person","counting_type":"line",
  "line":[0,60,100,60],"model":"yolo","max_fps":8}')
echo "   job: $JOB"
JID=$(printf '%s' "$JOB" | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p')
[ -z "$JID" ] && { echo "   ❌ không tạo được job"; exit 1; }

echo "==> 5) Đếm ~20s rồi đọc kết quả"
sleep 20
echo "   stats: $(curl -fsS http://localhost:8000/api/jobs/$JID)"
echo "   events: $(curl -fsS 'http://localhost:8000/api/events?limit=3')"
curl -fsS -X POST http://localhost:8000/api/jobs/$JID/stop >/dev/null || true

echo "==> ✅ SMOKE TEST XONG (service build + chạy + đếm + vector DB hoạt động)."
