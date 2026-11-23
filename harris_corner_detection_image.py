import cv2
import numpy as np

# 1. Load the source image
img = cv2.imread("harris_corner_detection_image.jpg")
if img is None:
    raise FileNotFoundError("Could not load harris_corner_detection_image.jpg - check the file path")

original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)  # cornerHarris requires float32

# 2. Harris parameters
block_size = 2      # neighborhood size considered for each pixel
aperture_size = 3   # Sobel operator aperture
k = 0.04            # Harris free parameter (0.04 - 0.06)

# 3. Detect corners
dst = cv2.cornerHarris(gray, block_size, aperture_size, k)

# 4. Dilate to make corner responses easier to see
dst = cv2.dilate(dst, None)

# 5. Threshold to keep only strong corners
threshold = 0.01 * dst.max()
ys, xs = np.where(dst > threshold)

# 6. Draw circles on corner locations
for x, y in zip(xs, ys):
    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), 1)

print(f"Corner pixels detected: {len(xs)}")

# 7. Display before / after
cv2.imshow("Original Image", original)
cv2.imshow("Harris Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
