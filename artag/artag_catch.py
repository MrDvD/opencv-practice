"""
Находит ArTag на листе бумаги и выводит его отдельным окном.
"""
import cv2 as cv
import numpy as np

def nothing(x):
    pass

cv.namedWindow('thresh')
white = 255 * np.ones([480, 640, 1], np.uint8)
black = np.zeros([480, 640, 1], np.uint8)
code = np.float32([[0, 0], [0, 300], [300, 300], [300, 0]])

cv.createTrackbar('h1', 'thresh', 0, 255, nothing)
cv.createTrackbar('s1', 'thresh', 0, 255, nothing)
cv.createTrackbar('v1', 'thresh', 0, 255, nothing)
cv.createTrackbar('h2', 'thresh', 255, 255, nothing)
cv.createTrackbar('s2', 'thresh', 155, 255, nothing)
cv.createTrackbar('v2', 'thresh', 48, 255, nothing)

cam = cv.VideoCapture(0)

cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    warped = None
    _, frame = cam.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    h1 = cv.getTrackbarPos('h1', 'thresh')
    s1 = cv.getTrackbarPos('s1', 'thresh')
    v1 = cv.getTrackbarPos('v1', 'thresh')
    h2 = cv.getTrackbarPos('h2', 'thresh')
    s2 = cv.getTrackbarPos('s2', 'thresh')
    v2 = cv.getTrackbarPos('v2', 'thresh')
    mask = cv.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
    result = cv.bitwise_and(white, mask)

    conts, _ = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in conts:
        approx = cv.approxPolyDP(i, 0.08 * cv.arcLength(i, True), True)
        if len(approx) == 4 and cv.contourArea(approx) > 3000:
            M = cv.getPerspectiveTransform(np.float32(approx), code)
            warped = cv.warpPerspective(result, M, (300, 300))
            warped = cv.bitwise_not(warped)

    cv.imshow('frame', frame)
    cv.imshow('thresh', result)
    if warped is not None:
        cv.imshow('code', warped)
    if cv.waitKey(1) == ord('q'):
        break