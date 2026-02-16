import cv2
import numpy as np
import os

print("OpenCV Version:", cv2.__version__)
print("Files in folder:", os.listdir())

# CHANGE IMAGE NAME TO EXISTING FILE (3.jpg or 4.jpg)
img = cv2.imread("4.jpg")

if img is None:
    print("❌ Image not loaded. Fix filename!")
    exit()

print("✅ Image loaded")

# Original
cv2.imshow("Original", img)
cv2.waitKey(0)

# Blur
blur = cv2.blur(img,(5,5))
cv2.imshow("Blur", blur)
cv2.waitKey(0)

# Bilateral
bilateral = cv2.bilateralFilter(img,9,75,75)
cv2.imshow("Bilateral", bilateral)
cv2.waitKey(0)

# HSV
HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", HSV_img)
cv2.waitKey(0)

cv2.destroyAllWindows()