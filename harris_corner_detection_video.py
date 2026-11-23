import cv2
import numpy as np

# Harris parameters
block_size = 2
aperture_size = 3
k = 0.04
threshold_ratio = 0.01

# Video file (use 0 instead for webcam)
cap = cv2.VideoCapture("harris_corner_detection_image_V.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open harris_corner_detection_image_V.mp4 - check the file path")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video

    original = frame.copy()
    gray = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    dst = cv2.cornerHarris(gray, block_size, aperture_size, k)
    dst = cv2.dilate(dst, None)

    ys, xs = np.where(dst > threshold_ratio * dst.max())
    for x, y in zip(xs, ys):
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), 1)

    cv2.imshow("Original Video", original)
    cv2.imshow("Harris Corners - Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
