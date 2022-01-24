#!/usr/bin/env python
import numpy as np
import cv2
from multiprocessing import Process, Queue

import rospy, sys
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
        # self.initialiseCamera()

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

        self.prev_data = None

        self.angle = self.AD.getAngle
        self.angularVelocity = self.AD.getAngularVelocity
        self.gripper_data = None
        self.last_move_time = 0

        self.PUBLISH = True

        while not (self.gripper_data or rospy.is_shutdown()):
            rospy.sleep(1)
            rospy.loginfo("Waiting for gripper!")

        self.q = Queue()

        self.p1 = Process(target=self.getCameraFrame)
        self.p2 = Process(target=self.control)

        self.p1.start()
        self.p2.start()

        # wait for p2 to finish
        self.p2.join()
        # kill the camera
        # self.p1.kill()

        while not rospy.is_shutdown():
            image = self.qRgb.get().getCvFrame()
            self.AD.update_angle(image)
            # print(round(self.angularVelocity(), 2), round(self.angle(), 2))

    # def initialiseCamera(self):


    def getCameraFrame(self):
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.create(dai.node.XLinkOut)

        xoutRgb.setStreamName("rgb")

        # Properties
        camRgb.setPreviewSize(300, 300)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setFps(60)

        camRgb.preview.link(xoutRgb.input)

        device = dai.Device(pipeline)
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)


        # bring camera in to focus
        for _ in range(20):
            img = qRgb.get().getCvFrame()

        # initialise things
        for _ in range(10):
            self.AD.update_angle(qRgb.get().getCvFrame())


        while not rospy.is_shutdown():
            image = qRgb.get().getCvFrame()
            self.AD.update_angle(image)
            self.q.put((timeit.default_timer(), self.angle(), self.angularVelocity()))

    def control(self):
        arrived = False
        t1 = timeit.default_timer()

        while not arrived and not rospy.is_shutdown():
            arrived = self.control_loop()

            t2 = timeit.default_timer()
            # print(
            #     round(1 / (t2 - t1), 2),
            #     round(self.angularVelocity(), 2),
            #     round(self.angle(), 2),
            # )
            t1 = timeit.default_timer()
        
        print("hello")

    def control_loop(self):

        # camera is not initalised
        if self.prev_data is None and self.q.empty():
            return False

        self.prev_data = self.prev_data if self.q.empty() else self.q.get()

        (prev_time, angle, angleVel) = self.prev_data

        time_since_update = max(0, prev_time - self.prev_data[0])
        time_delay = 0

        # future predict the angle of the object, offset by time delay
        object_angle = angle + angleVel * (time_delay + time_since_update)

        current_time = timeit.default_timer()

        if abs(self.goal - object_angle) < 1 or angle > self.goal:
            print(self.goal, angle, object_angle, "sent stopped!!")
            self.close_gripper()
            return True

        canMove = (current_time - self.last_move_time) > 0.05

        # if the object is slow, and the gripper has opened
        if angleVel < 0.3 and canMove:
            print("loooooosen")
            self.slightly_open_gripper()

        return False

    def slightly_open_gripper(self):
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rPR = self.gripper_width - 1
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

        self.last_move = timeit.default_timer()

        if self.PUBLISH:
            print(command)
            self.gripper_pub.publish(command)

    def gripper_state_callback(self, data):
        self.gripper_data = data
        self.gripper_width = data.gPO

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
