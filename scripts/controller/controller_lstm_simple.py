#!/usr/bin/env python3

from this import d
import cv2
import depthai as dai
import rospy
import rosbag
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Header, Float64
from papillarray_ros_v2.msg import SensorState

import time, timeit
import math
from cv_bridge import CvBridge
from digit_interface import Digit

from grasp_executor.msg import DataCollectState

import torch
from torch import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from blob_detector import AngleDetector
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg

import sys
sys.path.append('/home/acrv/new_ws/src/grasp_executor/scripts')
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg, gripper_position_msg
from enum import Enum

import pdb

class DataMode(Enum):
    POSITION = 0
    VELOCITY = 1
    BOTH = 2

class RegressionLSTM(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_layers, dropout, window_size, data_mode=None, seq_length=None):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # If both need 2 outputs for both position and velocity, else only need 1 output
        self.output_size = 2 if data_mode == DataMode.BOTH else 1
        print('Output size = %d' %(self.output_size))

        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=dropout
        )

        # self.output_linear = nn.Linear(
        #     in_features=self.hidden_size, out_features=self.hidden_size)
        self.output_linear_final = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size)

    def init_model_state(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).requires_grad_().to(self.device)
        return (h0, c0)

    def forward(self, x, h0=None, c0=None):
        batch_size = x.shape[0]

        if h0 is None or c0 is None:
            h0, c0 = self.init_model_state(batch_size)

        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # out = self.output_linear(out)
        out = self.output_linear_final(out)

        return out, h_n, c_n

def tactile_data_to_df(tac_0, tac_1): 
    cols_sensor = ['gfX', 'gfY', 'gfZ', 'gtX', 'gtY', 'gtZ', 'friction_est', 'target_grip_force']
    cols_pillar = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ', 'in_contact']

    l = []
    for i in range(2):
        tac = tac_0 if i==0 else tac_1
        weight = lambda attr : 34 if ('fX' in attr or 'fY' in attr or 'fZ' in attr) else 1
        for attr in cols_sensor:
                l.append(getattr(tac, attr))
        for j in range(9):
            for attr in cols_pillar:
                l.append(getattr(tac.pillars[j], attr))

    return np.array(l)



class DataBagger:
    def __init__(self):
        rospy.init_node("Data_bagger", anonymous=True)

        self.write_bags = False

        self.LSTM = RegressionLSTM("cpu", 142, 500, 3, 0.15, 0, DataMode.BOTH)

        self.AD = AngleDetector(writeImages=False, showImages=False, cv2Image=True)

        self.LSTM.load_state_dict(torch.load("best_model_nolin.pt"))
        self.LSTM.eval()

        self.h_n = None
        self.finish_angle = None
        self.c_n = None

        self.start_time = None
        self.lstm_started = False

        self.start_angle = None
        self.current_angle = None

        self.gripper_data = 0

        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        while not (self.gripper_data or rospy.is_shutdown()):
            rospy.sleep(1)
            print("Waiting for gripper!")

        self.gt_pub = rospy.Publisher('/gt_vel', Float64, queue_size=1)
        self.lstm_pub = rospy.Publisher('/lstm_vel', Float64, queue_size=1)

        self.gt_angle_pub = rospy.Publisher('/gt_ang', Float64, queue_size=1)
        self.lstm_angle_pub = rospy.Publisher('/lstm_ang', Float64, queue_size=1)

        self.use_papilarray = True
        self.digit_serials = ['D20235', 'D20226']

        self.image_topic = "/realsense/rgb"
        self.current_image = 0
        self.collect_data_flag = False
        self.collection_info = None
        # self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.collection_flag_sub = rospy.Subscriber('/collect_data', DataCollectState, self.collect_flag_callback)  ##TODO
        
        ##TODO: Node to read tactile data
        if self.use_papilarray:
            self.tactile_data_0 = (0, None)
            self.tactile_data_1 = (0, None)
                    
            self.tactile_0_sub = rospy.Subscriber('/hub_0/sensor_0', SensorState, self.sensor_0_callback)
            self.tactile_1_sub = rospy.Subscriber('/hub_0/sensor_1', SensorState, self.sensor_1_callback)
        else:
            d0 = Digit(self.digit_serials[0]) # Unique serial number
            d0.connect()
            d0.set_resolution(Digit.STREAMS["QVGA"])
            d0.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

            d1 = Digit(self.digit_serials[1]) # Unique serial number
            d1.connect()
            d1.set_resolution(Digit.STREAMS["QVGA"])
            d1.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

        bridge = CvBridge()

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

        # Linking
        prev_0 = 0
        prev_1 = 0

        camRgb.preview.link(xoutRgb.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            print('Connected cameras: ', device.getConnectedCameras())
            # Print out usb speed
            print('Usb speed: ', device.getUsbSpeed().name)

            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            t1 = timeit.default_timer()
            print("past qRGB")

            if self.use_papilarray:
                while self.tactile_data_0[1] == None or self.tactile_data_1[1] == None:
                    continue

            for _ in range(100):
                inRgb = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived

            print("Tactile running, node started")
            in_loop = False
            while not rospy.is_shutdown():
                
                if self.collect_data_flag: ##Start data collection

                    if self.write_bags:
                        if not in_loop:
                            bag0 = rosbag.Bag('./recorded_data_bags/data_'+str(int(math.floor(time.time())))+"_0.bag", 'w') 
                            bag1 = rosbag.Bag('./recorded_data_bags/data_'+str(int(math.floor(time.time())))+"_1.bag", 'w')
                            bag0.write('metadata', self.collection_info)
                            bag1.write('metadata', self.collection_info)
                            print('bag created')

                    in_loop = True
                    inRgb = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived
                    # print("got image")

                    if self.write_bags:
                        time_now = rospy.Time.from_sec(time.time())
                        header = Header()
                        header.stamp = time_now
                        cv_image = bridge.cv2_to_imgmsg(inRgb)

                    if prev_0 == self.tactile_data_0[1]:
                        print("uh oh repeated data")

                    if prev_1 == self.tactile_data_1[1]:
                        print("uh oh repeated data")

                    prev_0 = self.tactile_data_0[1]
                    prev_1 = self.tactile_data_1[1]

                    if self.write_bags:
                        bag0.write('time', header)
                        bag1.write('time', header)
                        bag0.write('image', cv_image) #Save an image
                        bag0.write('tactile_0', prev_0) #save forces
                        bag0.write('tactile_1', prev_1)


                    data = tactile_data_to_df(prev_0, prev_1)  
                    data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)
                    output, self.h_n, self.c_n, = self.LSTM(data, self.h_n, self.c_n)

                    self.AD.update_angle(inRgb)

                    self.start_time = self.start_time if self.start_time is not None else rospy.Time.now()

                    if not self.lstm_started and (rospy.Time.now() - self.start_time).to_sec() > 0.5:
                        self.lstm_started = True
                        self.command_gripper(gripper_position_msg(158))
                    elif not self.lstm_started:
                        print("skipping till 0.5s")
                    
                    gt_vel = self.AD.getAngularVelocity()
                    self.gt_pub.publish(gt_vel)

                    # pdb.set_trace()
                    
                    pred_vel = output[:,:,0].item() * 400
                    self.lstm_pub.publish(pred_vel)

                    pred_pos = output[:,:,1].item() * 90

                    self.lstm_angle_pub.publish(pred_pos)
                    gt_pos = self.AD.getAngle()
                    self.gt_angle_pub.publish(gt_pos)

                    if self.start_angle is None:
                        self.start_angle = gt_pos
                        self.current_angle = self.start_angle


                    use_gt = True
                    if use_gt:
                        self.current_angle = gt_pos #+ self.AD.startAngle
                        fwd_vel = gt_vel
                    else:
                        self.current_angle = pred_pos #+ self.AD.startAngle
                        fwd_vel = pred_vel

                    # future predict the angle of the object, offset by time delay
                    time_delay = 10/60
                    object_angle = self.current_angle + fwd_vel * time_delay

                    print('gt_angle: %.2f, fwd_angle: %.2f, curr_angle: %.2f' % (gt_pos, object_angle, self.current_angle))

                    goal = 60
                    if abs(object_angle - goal) < 5 or object_angle > goal:
                        print("************** CLOSING *************")
                        # print('  - object_angle = %.2f' % (object_angle))
                        self.close_gripper()
                        self.finish_angle = gt_pos
                        # self.collect_data_flag = False

                    if self.write_bags:
                        gt_vel_msg = Float64()
                        gt_vel_msg.data = gt_vel
                        bag1.write('gt_velocity', gt_vel_msg)

                        gt_pos_msg = Float64()
                        gt_pos_msg.data = gt_pos
                        bag1.write('gt_position', gt_pos_msg)

                        pred_vel_msg = Float64()
                        pred_vel_msg.data = pred_vel
                        bag1.write('pred_velocity', pred_vel_msg)

                        pred_pos_msg = Float64()
                        pred_pos_msg.data = pred_pos
                        bag1.write('pred_position', pred_pos_msg)


                    # print("object_angle: ", object_angle)
                    # print("lstm output: ", pred_vel)

                    # print("lstm angle: ", pred_pos)
                    # print("gt angle: ", self.AD.getAngle())
                    if self.finish_angle is not None:
                        print("finish angle: ", self.finish_angle)
                elif in_loop and self.collect_data_flag is False:
                    in_loop = False
                    ##TODO: restart everything
                    if self.write_bags:
                        if bag0 is not None:
                            bag0.close()
                        
                        if bag1 is not None:
                            bag1.close()

                    self.h_n = None
                    self.finish_angle = None
                    self.c_n = None

                    self.start_time = None
                    self.lstm_started = False

                    self.start_angle = None
                    self.current_angle = None

                    self.AD = AngleDetector(writeImages=False, showImages=False, cv2Image=True)

                    print("Reset!")



    def sensor_0_callback(self, sensor_state_msg):
        self.tactile_data_0 = (self.tactile_data_0[0] + 1, sensor_state_msg)

    def sensor_1_callback(self, sensor_state_msg):
        self.tactile_data_1 = (self.tactile_data_1[0] + 1, sensor_state_msg)

    def collect_flag_callback(self, bool_msg):
        self.collection_info = bool_msg
        self.collect_data_flag = bool_msg.data

    def close_gripper(self, width=255):
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rPR = width
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

        if True:
            self.gripper_pub.publish(command)

    def command_gripper(self, grip_msg):
        self.gripper_pub.publish(grip_msg)

    def gripper_state_callback(self, data):
        self.gripper_data = data
        
DataBagger()