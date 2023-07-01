"""
Рисует выбранным цветом в окне.
"""

import cv2 as cv
import numpy as np

def nothing(x):
   pass

class Draw:
   pressed = False
   
   def callback(event, x, y, *args):
      if event == cv.EVENT_LBUTTONDOWN:
         Draw.pressed = True
         Draw.paint(x, y)
      elif event == cv.EVENT_MOUSEMOVE:
         Draw.paint(x, y)
      elif event == cv.EVENT_LBUTTONUP:
         Draw.pressed = False
   
   def paint(x, y):
      if Draw.pressed:
         r, g, b = cv.getTrackbarPos('R', 'title2'), cv.getTrackbarPos('G', 'title2'), cv.getTrackbarPos('B', 'title2')
         cv.circle(empty, (x, y), 3, (b,g,r), 5)

empty = np.full((512,512,4), 255, np.uint8)

cv.namedWindow('title2')
cv.setMouseCallback('title2', Draw.callback)

cv.createTrackbar('R', 'title2', 0, 255, nothing)
cv.createTrackbar('G', 'title2', 0, 255, nothing)
cv.createTrackbar('B', 'title2', 0, 255, nothing)

while True:
   cv.imshow('title2', empty)
   key = cv.waitKey(1)
   if key == ord('q'):
      break

cv.destroyAllWindows()