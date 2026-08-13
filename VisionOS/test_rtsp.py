import cv2
cap = cv2.VideoCapture("rtsp://127.0.0.1:8554/traffic")
print("Opened:", cap.isOpened())
if cap.isOpened():
    ret, frame = cap.read()
    print("Frame shape:", frame.shape if ret else None)
