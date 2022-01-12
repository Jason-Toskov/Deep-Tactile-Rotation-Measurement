import numpy as np
import cv2

from enum import Enum

class Quadrant(Enum):
    NW = 0
    SW = 1
    SE = 2
    NE = 3

image = cv2.imread('img_temp.jpeg')
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0,85,0])
upper = np.array([7,255,255])
mask = cv2.inRange(image, lower, upper)
result = cv2.bitwise_and(result, result, mask=mask)

cv2.imshow('mask', mask)
cv2.imwrite('mask.jpeg', mask)
cv2.imshow('result', result)
cv2.imwrite('result.jpeg', result)
cv2.waitKey()

# params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.minArea = 100
# params.maxArea = 2000

# detector = cv2.SimpleBlobDetector_create()
# keypoints = detector.detect(mask)
# im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# blob = max(contours, key=lambda el: cv2.contourArea(el))
# print(type(contours))
contours.sort(key=lambda el: cv2.contourArea(el), reverse=True)
blob_list = contours
# print(blob_list)
blob = blob_list[0]
M = cv2.moments(blob)
center1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
# print(center)
canvas = result.copy()
cv2.circle(canvas, center1, 2, (0,255,0), -1)

blob2 = blob_list[1]
M = cv2.moments(blob2)
center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
# print(center)
cv2.circle(canvas, center2, 2, (0,255,0), -1)

# print(center1)
# print(center2)
# if center1[0] > center2[0]:
#     angle = np.arctan2(center2[1]-center1[1], center2[0]-center1[0])
# else:
#     angle = np.arctan2(center1[1]-center2[1], center1[0]-center2[0])

# print(angle*180/np.pi)

cv2.imshow('canvas', canvas)
cv2.imwrite('canvas.jpeg', canvas)

cv2.waitKey()
