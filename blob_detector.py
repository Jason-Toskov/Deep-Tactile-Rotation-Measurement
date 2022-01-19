#!/usr/bin/env python
import numpy as np
import cv2

import rospy
from time import sleep
from sensor_msgs.msg import Image
from std_msgs.msg import Empty as EmptyMsg
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import Empty, EmptyResponse
from grasp_executor.srv import AngleTrack

from enum import Enum

class Quadrant(Enum):
    INIT = 0
    NW = 1
    SW = 2
    SE = 3
    NE = 4


class AngleDetector():
    def __init__(self):
        rospy.init_node("Angle_detector")
        #self.image_topic = "/realsense/rgb"

        self.bridge = CvBridge()
        self.state = Quadrant.INIT
        self.closest_state = Quadrant.INIT
        self.angle = None

        #self.current_image = None
        #rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Service('track_angle', AngleTrack, self.update_angle)
        rospy.Service('reset_angle_tracking', Empty, self.reset_tracking)

        rospy.loginfo("Angle tracker ready!")
        rospy.spin()

    # def image_callback(self, data):
    #     self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def reset_tracking(self, req):
        self.state = Quadrant.INIT
        self.closest_state = Quadrant.INIT
        self.angle = None
        print('reset!')
        return EmptyResponse()

    def angle_calculation(self, point0, point1):
        if point0[0]-point1[0] == 0:
            temp_angle = 90
        else:
            temp_angle = np.arctan(np.abs(float(point0[1]-point1[1])/float(point0[0]-point1[0])))*180/np.pi

        # print((point0[1]-point1[1]))
        # print((point0[0]-point1[0]))
        # print(np.abs((point0[1]-point1[1])/(point0[0]-point1[0])))
        # print(self.state)

        if self.state == Quadrant.NW:
            self.angle = -temp_angle
        elif self.state == Quadrant.SW:
            self.angle = temp_angle
        elif self.state == Quadrant.SE:
            self.angle = 180 - temp_angle
        elif self.state == Quadrant.NE:
            self.angle = -180 + temp_angle

    def state_update(self, point0, point1):
        rightmost_point = 0 if point0[0] > point1[0] else 1
        lowest_point = 0 if point0[1] > point1[1] else 1

        if self.state == Quadrant.INIT:
            self.state = Quadrant.NW if rightmost_point == lowest_point else Quadrant.SW
            self.closest_state = Quadrant.SW if rightmost_point == lowest_point else Quadrant.NW
        elif self.state == Quadrant.NW:
            print(rightmost_point, lowest_point, self.state, self.closest_state)
            if rightmost_point is not lowest_point:
                self.state = self.closest_state
        elif self.state == Quadrant.SW:
            if rightmost_point is lowest_point:
                self.state = self.closest_state
        elif self.state == Quadrant.SE:
            if rightmost_point is not lowest_point:
                self.state = self.closest_state
        elif self.state == Quadrant.NE:
            if rightmost_point is lowest_point:
                self.state = self.closest_state


    def closest_new_state(self):
        self.closest_state = Quadrant.INIT
        if self.state == Quadrant.NW:
            self.closest_state = Quadrant.SW if self.angle > -45 else Quadrant.NE  
        elif self.state == Quadrant.SW:
            self.closest_state = Quadrant.NW if self.angle < 45 else Quadrant.SE
        elif self.state == Quadrant.SE:
            self.closest_state = Quadrant.SW if self.angle < 135 else Quadrant.NE
        elif self.state == Quadrant.NE:
            self.closest_state = Quadrant.NW if self.angle > -135 else Quadrant.SE


    def update_angle(self, req):
        # rate = rospy.Rate(0.5)

        # while self.current_image is None and not rospy.is_shutdown():
        #     rospy.sleep(1)

        # while not rospy.is_shutdown():

        print("Reached loop!")
        # image = cv2.imread('img_temp.jpeg')
        im = self.bridge.imgmsg_to_cv2(req.im, desired_encoding='8UC3')
        result = im.copy()
        image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        lower = np.array([0,85,0])
        upper = np.array([7,255,255])
        mask = cv2.inRange(image, lower, upper)


        lower1 = np.array([175,85,0])
        upper1 = np.array([180,255,255])
        mask1 = cv2.inRange(image, lower1, upper1)

        mask = mask + mask1

        result = cv2.bitwise_and(result, result, mask=mask)

        cv2.imshow('orig', im)
        cv2.imshow('hsv', image)
        cv2.imshow('mask', mask)

        cv2.imwrite('img_temp.jpeg', im)

        cv2.imwrite('mask.jpeg', mask)
        # cv2.imshow('result', result)
        cv2.imwrite('result.jpeg', result)

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

        # angle = self.angle_calculation(center1, center2)
        # print(angle)

        if self.angle is None:
            self.state_update(center1, center2)
            self.angle_calculation(center1, center2)
        else:
            self.closest_new_state()
            self.state_update(center1, center2)
            self.angle_calculation(center1, center2)

        print(self.angle)


        cv2.imshow('canvas', canvas)
        cv2.imwrite('canvas.jpeg', canvas)
        cv2.waitKey(1)


            # cv2.waitKey()

            # rate.sleep()

        return self.angle


if __name__ == "__main__":
    try:
        angle_class = AngleDetector()
    except KeyboardInterrupt:
        pass