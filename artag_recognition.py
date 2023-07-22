"""
Находит ArTag на листе бумаги, выводит его отдельным окном и сканирует с него информацию.
"""
import cv2 as cv
import numpy as np

def nothing(x):
    pass

cv.namedWindow('threshold')

cv.createTrackbar('h1', 'threshold', 0, 255, nothing)
cv.createTrackbar('s1', 'threshold', 0, 255, nothing)
cv.createTrackbar('v1', 'threshold', 0, 255, nothing)
cv.createTrackbar('h2', 'threshold', 255, 255, nothing)
cv.createTrackbar('s2', 'threshold', 155, 255, nothing)
cv.createTrackbar('v2', 'threshold', 48, 255, nothing)

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

font = cv.FONT_HERSHEY_COMPLEX

def get_value(image, x, y):
    block = image[y:y + 50, x:x + 50]
    return np.mean(block) < 128

white = 255 * np.ones([480, 640, 1], np.uint8)
code = np.float32([[0, 0], [0, 300], [300, 300], [300, 0]])

def decypher_code(image):
    ans, r, markers = '', -1, [(200, 200), (200, 50), (50, 50), (50, 200)]
    rotation = [cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    for i in range(4):
        x, y = markers[i]
        if not get_value(image, x, y):
            r = i
    if r == -1:
        return None, ''
    elif r:
        image = cv.rotate(image, rotation[r - 1])
    cv.putText(image, f'{r * 90} deg CW', (15, 35), font, 1, (255, 255, 255))

    for y in range(0, 300, 50):
        for x in range(0, 300, 50):
            if x in [0, 250] or y in [0, 250]:
                if not get_value(image, x, y):
                    return None, ''
                continue
            if (x, y) in markers:
                continue
            if get_value(image, x, y):
                ans += '1'
                cv.putText(image, '1', (x + 15, y + 35), font, 1, (255, 255, 255))
            else:
                ans += '0'
                cv.putText(image, '0', (x + 15, y + 35), font, 1, (0, 0, 0))
    return image, ans

timer, ans = 0, ''
while True:
    if timer > 60 and ans:
        break

    scan_area = None
    _, frame = cam.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    h1 = cv.getTrackbarPos('h1', 'threshold')
    s1 = cv.getTrackbarPos('s1', 'threshold')
    v1 = cv.getTrackbarPos('v1', 'threshold')
    h2 = cv.getTrackbarPos('h2', 'threshold')
    s2 = cv.getTrackbarPos('s2', 'threshold')
    v2 = cv.getTrackbarPos('v2', 'threshold')
    
    mask = cv.inRange(hsv, (h1, s1, v1), (h2, s2, v2))
    thresh = cv.bitwise_and(white, mask)

    conts, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in conts:
        approx = cv.approxPolyDP(c, 0.08 * cv.arcLength(c, True), True)
        if len(approx) == 4 and cv.contourArea(approx) > 3000:
            transform = cv.getPerspectiveTransform(np.float32(approx), code)
            warped = cv.warpPerspective(thresh, transform, (300, 300))
            scan_area = cv.bitwise_not(warped)

    cv.imshow('frame', frame)
    cv.imshow('threshold', thresh)
    if scan_area is not None:
        artag, curr_ans = decypher_code(scan_area)
        if artag is not None:
            cv.imshow('code', artag)
            if ans != curr_ans:
                ans = curr_ans
                timer = 0
    else:
        timer = 0
    
    timer += 1

    if cv.waitKey(1) == ord('q'):
        break
print('=====================')
print('Code:', ans)