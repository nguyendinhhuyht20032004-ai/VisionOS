@echo off
:: =========================================================================
:: run_native.bat -- Chay VisionOS AI Service NATIVE tren Windows
:: =========================================================================

:: --- Model weights ---
if "%YOLO_WEIGHTS%"=="" set YOLO_WEIGHTS=yolov8m_openvino_model

:: Kiem tra thu muc model OpenVINO
if "%YOLO_WEIGHTS:~-15%"=="_openvino_model" (
    if not exist "%YOLO_WEIGHTS%" (
        echo.
        echo Chua tim thay thu muc %YOLO_WEIGHTS%
        echo Chay export truoc bang lenh:
        echo   python export_openvino.py --weights yolov8m.pt
        echo.
        echo Hoac dung PyTorch ^(cham hon^):
        echo   set YOLO_WEIGHTS=yolov8m.pt ^&^& run_native.bat
        exit /b 1
    )
)

:: --- Environment variables ---
if "%CONTROL_API_PORT%"=="" set CONTROL_API_PORT=8000
if "%OVERLAY_GRPC_TARGET%"=="" set OVERLAY_GRPC_TARGET=127.0.0.1:9090
if "%YOLO_IMGSZ%"=="" set YOLO_IMGSZ=640
if "%YOLO_CONF%"=="" set YOLO_CONF=0.3
if "%DETECT_EVERY%"=="" set DETECT_EVERY=1
if "%OVERLAY_PUBLISH_FPS%"=="" set OVERLAY_PUBLISH_FPS=30
if "%SMOOTHER_LEN%"=="" set SMOOTHER_LEN=2
if "%REID_STITCH%"=="" set REID_STITCH=0
if "%RTSP_TRANSPORT%"=="" set RTSP_TRANSPORT=tcp
if "%RTSP_RECONNECT_INTERVAL_SEC%"=="" set RTSP_RECONNECT_INTERVAL_SEC=5
if "%DEBUG_SHOW%"=="" set DEBUG_SHOW=1

echo.
echo ==========================================
echo   VisionOS AI Service ^(Native Windows^)
echo ==========================================
echo   Model:       %YOLO_WEIGHTS%
echo   ImgSz:       %YOLO_IMGSZ%
echo   Conf:        %YOLO_CONF%
echo   gRPC Target: %OVERLAY_GRPC_TARGET%
echo   Port:        %CONTROL_API_PORT%
echo   DetectEvery: %DETECT_EVERY%
echo   PublishFPS:  %OVERLAY_PUBLISH_FPS%
echo ==========================================
echo.

:: --- Tao thu muc data neu chua co ---
if not exist "data\videos" mkdir "data\videos"

:: --- Chay service ---
python run_service.py --host 127.0.0.1 --port %CONTROL_API_PORT%
