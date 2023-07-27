"""
Определение занятости ячейки статичного изображения.
"""
import cv2 as cv
import numpy as np

def nothing(x):
    pass

cv.namedWindow('camera')
cv.namedWindow('thresh_cells')

font = cv.FONT_HERSHEY_COMPLEX

cv.createTrackbar('h1', 'camera', 148, 255, nothing)
cv.createTrackbar('s1', 'camera', 150, 255, nothing)
cv.createTrackbar('v1', 'camera', 0, 255, nothing)
cv.createTrackbar('h2', 'camera', 255, 255, nothing)
cv.createTrackbar('s2', 'camera', 255, 255, nothing)
cv.createTrackbar('v2', 'camera', 196, 255, nothing)
cv.createTrackbar('kernel', 'camera', 7, 255, nothing)

while True:
    image = cv.imread('images/map2.png')

    # binarization
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h1 = cv.getTrackbarPos('h1', 'camera')
    s1 = cv.getTrackbarPos('s1', 'camera')
    v1 = cv.getTrackbarPos('v1', 'camera')
    h2 = cv.getTrackbarPos('h2', 'camera')
    s2 = cv.getTrackbarPos('s2', 'camera')
    v2 = cv.getTrackbarPos('v2', 'camera')
    res = cv.inRange(image, (h1, s1, v1), (h2, s2, v2))
    
    # noise removal
    kernel = cv.getTrackbarPos('kernel', 'camera')
    if not kernel:
        kernel = 1
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel, kernel))
    res = cv.morphologyEx(res, cv.MORPH_OPEN, kernel)

    conts, _ = cv.findContours(res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    texts = list()
    for c in conts:
        # rectangle test
        x, y, w, h = cv.boundingRect(c)
        if not w * h > 500:
            continue
        if not 0.6 < w / h < 1.4:
            continue
        # find aruco marker
        cell = image[y - 6:y + h + 6, x - 6:x + w + 6]

        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        params = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(dictionary, params)
        marks, ids, _ = detector.detectMarkers(cell)
        if marks:
            colour = (0, 0, 255)
            busy_id = f'id:{ids[0][0]}'
            texts += [(x, y, busy_id)]
        else:
            colour = (0, 255, 0)
        cv.rectangle(image, (x, y), (x + w, y + h), colour)
    # add ids
    for i in texts:
        cv.putText(image, i[2], (i[0], i[1] - 6), font, 0.4, (0, 255, 255))

    cv.imshow('camera', image)
    cv.imshow('thresh_cells', res)

    if cv.waitKey(1) == ord('q'):
        break