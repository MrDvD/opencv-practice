"""
Считывает только зелёные и синие предметы с камеры.
"""

import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)

colour_to_track = np.uint8([[[20,110,60]]])
hsv_color = cv.cvtColor(colour_to_track, cv.COLOR_RGB2HSV)
print(hsv_color)

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    blue_l = np.array([92, 100, 100])
    blue_r = np.array([112, 255, 255])
    green_l = np.array([63, 65, 65])
    green_r = np.array([83, 255, 255])
    blue = cv.inRange(hsv, blue_l, blue_r)
    green = cv.inRange(hsv, green_l, green_r)
    res_b = cv.bitwise_and(frame, frame, mask=blue)
    res_g = cv.bitwise_and(frame, frame, mask=green)
    res = cv.add(res_b, res_g)

    cv.imshow('g', res)
    if cv.waitKey(1) == ord('q'):
        break