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
import depthai as dai
from robotiq_2f_gripper_control.msg import (
    _Robotiq2FGripper_robot_output as outputMsg,
    _Robotiq2FGripper_robot_input as inputMsg,
)
import timeit

from blob_detector import AngleDetector


class AdaptiveController:
    def __init__(self):
        rospy.init_node("Controller")

        self.AD = AngleDetector(writeImages=False, showImages=False, cv2Image=True)
        self.goal = 60
        self.initialiseCamera()

        self.gripper_sub = rospy.Subscriber(
            "/Robotiq2FGripperRobotInput",
            inputMsg.Robotiq2FGripper_robot_input,
            self.gripper_state_callback,
        )
        self.gripper_pub = rospy.Publisher(
            "/Robotiq2FGripperRobotOutput",
            outputMsg.Robotiq2FGripper_robot_output,
            queue_size=1,
        )


        # bring camera in to focus
        for _ in range(20):
            img = self.qRgb.get().getCvFrame()
            
        # initialise things
        for _ in range(10):
            self.AD.update_angle(self.qRgb.get().getCvFrame())

        self.angle = self.AD.getAngle
        self.angularVelocity = self.AD.getAngularVelocity
        self.gripper_data = None
        self.last_move_time = 0

        self.PUBLISH = True

        while not (self.gripper_data or rospy.is_shutdown()):
            rospy.sleep(1)
            rospy.loginfo("Waiting for gripper!")

        arrived = False
        t1 = timeit.default_timer()
        while not arrived and not rospy.is_shutdown():
            arrived = self.control_loop()
            t2 = timeit.default_timer()  
            print(round(1 / (t2 - t1),2), round(self.angularVelocity(),2), round(self.angle(), 2))
            t1 = timeit.default_timer()

        while not rospy.is_shutdown():
            image = self.qRgb.get().getCvFrame()
            self.AD.update_angle(image)
            print(round(self.angularVelocity(),2), round(self.angle(), 2))

        print("reached goal!!")

    def initialiseCamera(self):

        self.pipeline = dai.Pipeline()

        # Define source and output
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)

        self.xoutRgb.setStreamName("rgb")

        # Properties
        self.camRgb.setPreviewSize(300, 300)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        self.camRgb.setFps(60)

        self.camRgb.preview.link(self.xoutRgb.input)

        self.device = dai.Device(self.pipeline)
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)


    def control_loop(self):
        time_delay = 0.05

        image = self.qRgb.get().getCvFrame()
        self.AD.update_angle(image)

        # future predict the angle of the object, offset by time delay
        object_angle = self.angle() + self.angularVelocity() * time_delay

        current_time = timeit.default_timer()

        if abs(self.goal - object_angle) < 1 or self.angle() > self.goal:
            print("sent stopped!!")
            self.close_gripper()
            return True

        canMove = (current_time - self.last_move_time) > 0.1

        # if the object is slow, and the gripper has opened
        if self.angularVelocity() < 0.3 and canMove:
            self.slightly_open_gripper()

        return False

    def slightly_open_gripper(self):
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rPR = self.gripper_width - 1
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150
        print("open a lil!!")

        self.last_move = timeit.default_timer()
        
        if self.PUBLISH:
            self.gripper_pub.publish(command)

    def gripper_state_callback(self, data):
        self.gripper_data = data
        self.gripper_width = data.gPR

    def close_gripper(self, width=255):
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rPR = width
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

        if self.PUBLISH:
            self.gripper_pub.publish(command)

if __name__ == "__main__":
    a = AdaptiveController()
    rospy.spin()