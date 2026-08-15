import grpc
from concurrent import futures
import time
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Đảm bảo đường dẫn import đúng
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recognition.service.proto import overlay_pb2, overlay_pb2_grpc

class MockBackendIngest(overlay_pb2_grpc.OverlayIngestServicer):
    def Health(self, request, context):
        print("[Mock Backend] Nhận request Health Check -> Phản hồi: ready=True")
        return overlay_pb2.HealthResponse(ready=True)

    def PublishFrames(self, request_iterator, context):
        print("[Mock Backend] Nhận kết nối stream Bounding Box từ AI Service...")
        count = 0
        try:
            for frame in request_iterator:
                count += 1
                if count % 10 == 0:  # In log mỗi 10 frame để tránh trôi màn hình quá nhanh
                    print(f"\n[Mock Backend] Nhận Frame #{count} | Stream: '{frame.job_key}' | Số lượng Object: {len(frame.boxes)}")
                    for i, box in enumerate(frame.boxes):
                        print(f"   => Object {i+1}: TrackID=[{box.track_id}] Class=[{box.class_name}] Conf={box.confidence:.2f} | Tọa độ: (x={box.x:.1f}, y={box.y:.1f}, w={box.w:.1f}, h={box.h:.1f})")
        except Exception as e:
            print(f"[Mock Backend] Stream bị ngắt kết nối: {e}")
        
        return overlay_pb2.PublishSummary(received=count, dropped=0)

    def PublishEvent(self, request, context):
        kind_str = "START"
        if request.kind == overlay_pb2.DetectionEvent.IN:
            kind_str = "IN (Vào vùng/Cắt vạch)"
        elif request.kind == overlay_pb2.DetectionEvent.OUT:
            kind_str = "OUT (Ra khỏi vùng)"
            
        print(f"\n[Mock Backend] NHAN SU KIEN QUAN TRONG")
        print(f"  - Event ID: {request.event_id}")
        print(f"  - Stream: {request.job_key} | Camera: {request.camera_id}")
        print(f"  - Track ID: {request.track_id} | Object: {request.class_name}")
        print(f"  - Loại sự kiện: {kind_str}\n")
        
        return overlay_pb2.EventAck(accepted=True)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    overlay_pb2_grpc.add_OverlayIngestServicer_to_server(MockBackendIngest(), server)
    
    # Lắng nghe ở cổng 9090 (cổng mặc định AI Service đang tìm kiếm)
    server.add_insecure_port('0.0.0.0:9090')
    server.start()
    print("="*60)
    print("[Mock Backend] dang chay tai 127.0.0.1:9090")
    print("Lắng nghe gRPC từ AI Service...")
    print("="*60)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nĐã dừng Mock Backend.")

if __name__ == '__main__':
    serve()
