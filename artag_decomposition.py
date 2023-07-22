"""
Раскладывает ArTag на пиксели и преобразует в матрицу.
"""
import cv2 as cv
import numpy as np

artag = cv.imread('images/artag2.png')
font = cv.FONT_HERSHEY_COMPLEX

def get_corners(image):
    corners = list()
    for x in [0, image.shape[1]]:
        for y in [0, image.shape[0]]:
            corners.append((x, y))
    return np.float32(sorted(corners))

corns1 = get_corners(artag)
corns2 = get_corners(np.zeros((300, 300), np.uint8))
transform = cv.getPerspectiveTransform(corns1, corns2)
scan_area = cv.warpPerspective(artag, transform, (300, 300))

def get_value(image, x, y):
    block = image[y:y + 50, x:x + 50]
    _, thresh = cv.threshold(block, 127, 255, cv.THRESH_BINARY)
    return int(np.all(thresh[0, 0] == 0))

code, markers = '', [(200, 200), (200, 50), (50, 50), (50, 200)]
rotation = [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
for i in range(4):
    x, y = markers[i]
    if not get_value(scan_area, x, y):
        r = i
if r:
    scan_area = cv.rotate(scan_area, rotation[r - 1])
cv.putText(scan_area, f'{r * 90} deg CW', (15, 35), font, 1, (0, 255, 255))

for y in range(50, 250, 50):
    for x in range(50, 250, 50):
        if (x, y) in markers:
            continue
        if get_value(scan_area, x, y):
            code += '1'
            cv.putText(scan_area, '1', (x + 15, y + 35), font, 1, (0, 255, 255))
        else:
            code += '0'
            cv.putText(scan_area, '0', (x + 15, y + 35), font, 1, (0, 0, 255))

cv.namedWindow('code')
cv.imshow('code', scan_area)

print(code)

while True:
    if cv.waitKey(1) == ord('q'):
        break