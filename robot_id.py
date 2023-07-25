"""
Сканирует маркеры и сопоставляет id, ориентацию роботов.
"""
import cv2 as cv
import numpy as np

class Robot:
    def __init__(self, num, orientation):
        self.id = num
        self.orientation = orientation
    
    def direction(func):
        def decorator(*args, **kwargs):
            orientation = func(*args, **kwargs)
            if orientation == 0:
                return 'up'
            if orientation == 1:
                return 'right'
            if orientation == 2:
                return 'down'
            if orientation == 3:
                return 'left'
        return decorator

    @direction
    def check_orientation(self, seq):
        for i in range(4):
            if seq[i:] + seq[:i] == self.orientation:
                return i
        return None

class Colour:
    def __init__(self, b, g, r, point):
        self.b, self.g, self.r = b, g, r
        self.point = point
    
    def colour(self):
        if abs(self.b - self.g) < 17 and abs(self.r - self.g) < 17 and \
           abs(self.r - self.b) < 17:
            return 'w'
        elif self.b > self.g and self.b > self.r:
            return 'b'
        elif self.g > self.b and self.b > self.r:
            return 'g'
        elif self.r > self.b and self.r > self.g:
            return 'r'

def nothing(x):
    pass

font = cv.FONT_HERSHEY_COMPLEX
robot = cv.imread('images/robot.png')

cv.namedWindow('result')
cv.namedWindow('thresh')
cv.namedWindow('robot')

cv.createTrackbar('blocksize', 'robot', 27, 255, nothing)
cv.createTrackbar('const', 'robot', 1, 255, nothing)
cv.createTrackbar('blur', 'robot', 3, 255, nothing)
cv.createTrackbar('kernel', 'robot', 4, 255, nothing)

cv.createTrackbar('blocksize', 'thresh', 115, 255, nothing)
cv.createTrackbar('const', 'thresh', 13, 255, nothing)
cv.createTrackbar('blur', 'thresh', 0, 255, nothing)

database = {(2, 2, 1): Robot(4, 'brgg'), (1, 2, 2): Robot(7, 'gbrr')}

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

def get_trackbar(window):
    blocksize = cv.getTrackbarPos('blocksize', window)
    if not blocksize % 2:
        blocksize += 1
    if blocksize <= 1:
        blocksize = 3
    const = cv.getTrackbarPos('const', window)
    blur = cv.getTrackbarPos('blur', window)
    if not blur % 2:
        blur += 1
    if blur <= 1:
        blur = 3
    return blocksize, const, blur

def get_threshold(image, blocksize, const, blur, invert=False, kernel=None):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    adapt = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                 cv.THRESH_BINARY, blocksize, const)
    if invert:
        adapt = cv.bitwise_not(adapt)
    if kernel is not None:
        adapt = cv.morphologyEx(adapt, cv.MORPH_OPEN, kernel)
    return cv.medianBlur(adapt, blur)

def is_circle(cont, min_area, ratio_prec, bounded=None):
    area = cv.contourArea(cont)
    _, _, width, height = cv.boundingRect(cont)
    hull = cv.convexHull(cont)
    hull_area = cv.contourArea(hull)
    if bounded is not None and area / cv.contourArea(bounded) > 0.1:
        return False
    return area > min_area and 1 - ratio_prec < width / height < 1 + ratio_prec and \
           0.9 < hull_area / area < 1.1

def is_inner(outer, cont):
    x_out, y_out, w_out, h_out = cv.boundingRect(outer)
    x_in, y_in, w_in, h_in = cv.boundingRect(cont)
    bound_out = np.array([(x_out, y_out), (x_out, y_out + h_out), \
                          (x_out + w_out, y_out + h_out), (x_out + w_out, y_out)])
    bound_in = np.array([(x_in, y_in), (x_in, y_in + h_in), \
                         (x_in + w_in, y_in + h_in), (x_in + w_in, y_in)])
    for p in bound_in:
        if cv.pointPolygonTest(bound_out, (int(p[0]), int(p[1])), False) != 1:
            return False
    return True

def get_colour_bgr(image, cont):
    mask = np.zeros((480, 640), np.uint8)
    cv.drawContours(mask, cont, -1, 255, -1)
    return cv.mean(image, mask)

def show_colours_bgr(colours, image, x, y):
    count = {'b': 0, 'g': 0, 'r': 0, 'w': 0}
    for c in colours:
        if c.colour():
            count[c.colour()] += 1
    code = (count['b'], count['g'], count['r'])
    if len(colours) != 5:
        return False
    elif code in database:
        x_sort = sorted(colours, key=lambda x: x.point[0])
        y_sort = sorted(colours, key=lambda x: x.point[1])
        seq = y_sort[0].colour() + x_sort[-1].colour() + y_sort[-1].colour() + x_sort[0].colour()
        orientation = database[code].check_orientation(seq)
        if orientation is None:
            return False
        info = f'id: {database[code].id} [{orientation}]'
    else:
        info = 'data_err'
    cv.putText(image, info, (x, y - 10), font, 0.8, (255, 255, 255))
    return True

def scan_code(image, offset, bound, blocksize, const, blur, invert=False):
    kernel = cv.getTrackbarPos('kernel', 'robot')
    if kernel < 1:
        kernel = 1
    kernel = np.ones((kernel, kernel), np.uint8)
    blurred_ins = get_threshold(image, blocksize, const, blur, invert, kernel)
    conts, _ = cv.findContours(blurred_ins, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, \
                               offset=offset)
    colours = list()
    idx = 1
    for c in conts:
        if not is_inner(bound, c):
            continue
        if is_circle(c, 100, 0.6, bound):
            x, y, _, _ = cv.boundingRect(c)
            mean = get_colour_bgr(frame, c)
            # debug = ' '.join(map(lambda x: str(round(x)), mean[:3]))
            # cv.putText(frame, str(idx), (x, y), font, 0.4, (255, 255, 255))
            idx += 1
            colours.append(Colour(*mean[:3], (x, y)))
            # cv.drawContours(frame, [c], -1, (0, 255, 255), 2)
    if show_colours_bgr(colours, frame, *offset):
        cv.drawContours(frame, [bound], -1, (0, 0, 255), 2)
    return blurred_ins

thresh = None
while True:
    _, frame = cam.read()
    blocksize1, const1, blur1 = get_trackbar('thresh')
    blurred = get_threshold(frame, blocksize1, const1, blur1, True)

    conts, _ = cv.findContours(blurred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in conts:
        if is_circle(c, 500, 0.2):
            blocksize, const, blur = get_trackbar('robot')
            x, y, w, h = cv.boundingRect(c)
            circle = frame[y:y + h, x:x + w]
            thresh = scan_code(circle, (x, y), c, blocksize, const, blur)
    
    cv.imshow('result', frame)
    cv.imshow('thresh', blurred)
    if thresh is not None:
        cv.imshow('robot', thresh)
    
    if cv.waitKey(1) == ord('q'):
        break