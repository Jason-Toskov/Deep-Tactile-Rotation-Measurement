#!/usr/bin/env python3

import cv2
import depthai as dai
import rospy
import rosbag
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from papillarray_ros_v2.msg import SensorState

import time, timeit
import math, numpy as np
from cv_bridge import CvBridge
import pdb
from scipy.spatial.transform import Rotation as R

class DataBagger:
    def __init__(self):
        rospy.init_node("Data_bagger", anonymous=True)

        bridge = CvBridge()

        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.create(dai.node.ColorCamera)
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        controlIn = pipeline.create(dai.node.XLinkIn)

        controlIn.setStreamName('control')
        controlIn.out.link(camRgb.inputControl)

        xoutRgb.setStreamName("rgb")

        # Properties
        camRgb.setPreviewSize(1920, 1080)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setFps(60)
        

        # print(dir(camRgb.properties))
        # print(dir(camRgb))

        camRgb.preview.link(xoutRgb.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            print('Connected cameras: ', device.getConnectedCameras())
            # Print out usb speed
            print('Usb speed: ', device.getUsbSpeed().name)

            # print(device.getConnectedCameras())
            camid = device.getConnectedCameras()[0]
            calib = device.readCalibration()
            matrix = calib.getDefaultIntrinsics(camid)
            dCoeff = calib.getDistortionCoefficients(camid)
            # pdb.set_trace()
            print(matrix, dCoeff)

            controlQueue = device.getInputQueue('control')
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(132)
            controlQueue.send(ctrl)


            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            while True:
                inRgb = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived

                arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)

                arucoParams = cv2.aruco.DetectorParameters_create()
                (corners, ids, rejected_img_points) = cv2.aruco.detectMarkers(inRgb, arucoDict,
                    parameters=arucoParams)
                cv2.aruco.drawDetectedMarkers(inRgb, corners, ids)

                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(inRgb, arucoDict)
                # print(matrix[0])
                rvec, tvec, x = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, np.array(matrix[0]), np.array(dCoeff[:-2]))
                if rvec is not None and len(rvec) > 0:
                    cv2.aruco.drawAxis(inRgb, np.array(matrix[0]), np.array(dCoeff[:-2]), rvec[0], tvec[0], 0.1)
                    r_mat, _ = cv2.Rodrigues(rvec)
                    rotation = R.from_matrix(r_mat)
                    print("rotation", rotation.as_euler("xyz", degrees=True))
                    cv2.imshow("rgb", inRgb)
                    cv2.waitKey(0)
                else:

                    for rejected in rejected_img_points:
                        rejected = rejected.reshape((4, 2))
                        cv2.line(inRgb, tuple(rejected[0]), tuple(rejected[1]), (0, 0, 255), thickness=2)
                        cv2.line(inRgb, tuple(rejected[1]), tuple(rejected[2]), (0, 0, 255), thickness=2)
                        cv2.line(inRgb, tuple(rejected[2]), tuple(rejected[3]), (0, 0, 255), thickness=2)
                        cv2.line(inRgb, tuple(rejected[3]), tuple(rejected[0]), (0, 0, 255), thickness=2)

                    cv2.imshow("rgb", inRgb)
                    cv2.waitKey(1)

                # print(rvec, corners, ids)

DataBagger()