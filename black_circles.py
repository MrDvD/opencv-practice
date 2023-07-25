"""
Находит чёрные круги.
"""
import cv2 as cv
import numpy as np

def nothing(x):
    pass

cv.namedWindow('camera')
cv.namedWindow('adaptive')
cv.namedWindow('blurred')

cv.createTrackbar('blocksize', 'adaptive', 140, 255, nothing)
cv.createTrackbar('C', 'adaptive', 18, 255, nothing)

cv.createTrackbar('ksize', 'blurred', 6, 255, nothing)

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

white = 255 * np.ones((480, 640), np.uint8)

while True:
    _, frame = cam.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blocksize = cv.getTrackbarPos('blocksize', 'adaptive')
    if not (blocksize % 2):
        blocksize += 1
    if blocksize <= 1:
        blocksize = 3
    c = cv.getTrackbarPos('C', 'adaptive')
    adapt = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                 cv.THRESH_BINARY, blocksize, c)
    adapt = cv.bitwise_not(adapt)

    ksize = cv.getTrackbarPos('ksize', 'blurred')
    if not (ksize % 2):
        ksize += 1
    if ksize <= 1:
        ksize = 3
    blurred = cv.medianBlur(adapt, ksize)

    conts, _ = cv.findContours(blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in conts:
        area = cv.contourArea(c)
        x, y, w, h = cv.boundingRect(c)
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        if area > 500 and 0.8 < w / h < 1.2 and 0.95 < hull_area / area < 1.05:
            cv.drawContours(frame, [c], -1, (0, 255, 255), 2)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0))
    
    cv.imshow('camera', frame)
    cv.imshow('adaptive', adapt)
    cv.imshow('blurred', blurred)
    if cv.waitKey(1) == ord('q'):
        break