#!/usr/bin/env python3
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import math
 
# Load the image
# img = cv.imread("input_img.jpg")
image = cv.imread("../images/tape2.png")

result = image.copy()
image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

lower = np.array([155,50,50])
upper = np.array([179,255,255])
mask1 = cv.inRange(image, lower, upper)

lower = np.array([0,150,70])
upper = np.array([179,255,255])
mask2 = cv.inRange(image, lower, upper)

mask = mask1+mask2

img = cv.bitwise_and(result, result, mask=mask)
# ---------- experimental (END) -----------
 
# Was the image there?
# if img is None:
#   print("Error: File not found")
#   exit(0)
 
# cv.imshow('Input Image', img)
 
# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
# Convert image to binary
_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
 
# Find all the contours in the thresholded image
contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
 
for i, c in enumerate(contours):
 
  # Calculate the area of each contour
  area = cv.contourArea(c)
  # print(area)
 
  # Ignore contours that are too small or too large
  if area < 2000 or 100000 < area:
    continue
 
  # cv.minAreaRect returns:
  # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
  rect = cv.minAreaRect(c)
  box = cv.boxPoints(rect)
  print(box)
  box = np.int0(box)
 
  # Retrieve the key parameters of the rotated bounding box
  center = (int(rect[0][0]),int(rect[0][1])) 
  width = int(rect[1][0])
  height = int(rect[1][1])
  angle = int(rect[2])
  
  shape = "unidentified"
  ar = width / float(height)
  # shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
   
  if width < height:
    angle = 90 - angle
  else:
    angle = -angle

  angle = math.radians(angle)

  if ar >= 0.95 and ar <= 1.05:
    shape = "square"
    angle = 0
  else:
    shape = "rectangle"
         
  # label = " Angle: " + str(angle) + " deg,"+str(center)
  label = " Angle: " + str(angle) + " deg,"+str(shape)
  textbox = cv.rectangle(img, (center[0]-35, center[1]-25), 
    (center[0] + 295, center[1] + 10), (255,255,255), -1)
  cv.putText(img, label, (center[0]-50, center[1]), 
    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv.LINE_AA)
  cv.drawContours(img,[box],0,(0,0,255),2)
 
# cv.imshow('Output Image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
  
# Save the output image to the current directory
cv.imwrite("../images/min_area_rec_output.jpg", img)