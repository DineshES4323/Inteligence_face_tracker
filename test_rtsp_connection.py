import cv2
import getpass

url = getpass.getpass("Enter RTSP URL (input hidden): ").strip()

if not url:
    print("ERROR: RTSP URL is empty.")
    raise SystemExit(1)

print("Connecting to RTSP stream...")
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("FAILED: Could not open RTSP stream.")
    cap.release()
    raise SystemExit(1)

print("CONNECTED: RTSP stream opened successfully.")

success_count = 0

for i in range(10):
    ret, frame = cap.read()

    if ret and frame is not None and frame.size > 0:
        success_count += 1
        print(f"Frame {i + 1}: OK - {frame.shape[1]}x{frame.shape[0]}")
    else:
        print(f"Frame {i + 1}: FAILED")

cap.release()

print(f"\nSuccessful frames: {success_count}/10")

if success_count == 10:
    print("RESULT: Basic RTSP connection test PASSED.")
else:
    print("RESULT: Frame reading needs investigation.")
