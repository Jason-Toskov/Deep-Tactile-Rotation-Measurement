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

from grasp_executor.msg import DataCollectState, ControllerData

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

from filterpy.kalman import KalmanFilter

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

        self.output_linear = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size)
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

        out = self.output_linear(out)
        out = self.output_linear_final(out)

        return out, h_n, c_n

class WindowMLP(nn.Module):
    def __init__(self, device, num_features, hidden_size, num_layers, dropout, window_size, data_mode=None, seq_length=None):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.seq_length = seq_length
        self.window_size = window_size
        self.output_size = 2 if data_mode == DataMode.BOTH else 1

        self.in_size = self.num_features*self.window_size

        self.lin1 = nn.Linear(self.in_size, self.in_size//8)
        self.lin2 = nn.Linear(self.in_size//8, self.in_size//16)
        self.lin3 = nn.Linear(self.in_size//16, self.in_size//32)
        self.lin4 = nn.Linear(self.in_size//32, self.output_size)

        self.drop = nn.Dropout(p=dropout)
        self.act = nn.Tanh()

        self.last_row = torch.zeros((1,self.output_size))

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.drop(self.act(self.lin1(x)))
        x = self.drop(self.act(self.lin2(x)))
        x = self.drop(self.act(self.lin3(x)))
        out = self.drop(self.act(self.lin4(x)))

        mask = torch.ne(out, torch.zeros(out.shape).to(self.device))
        self.last_row[mask] = out[mask]

        return self.last_row


class ModelVelocityTracker:
    def __init__(self):
        self.reset_tracking()

    def reset_tracking(self):
        self.f = KalmanFilter (dim_x=2, dim_z=1)

        self.initial_state_set = False

        # transition matrix
        self.f.F = np.array([[1.,1/60],
            [0.,1.]])

        # measurement function
        self.f.H = np.array([[1.,0.]])

        # covairance function
        self.f.P *= 0.00001
        # low measurement noise
        self.f.R = 0.00001

        from filterpy.common import Q_discrete_white_noise
        self.f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

        self.angle = 0
        self.angular_velocity = 0
        self.calc_time = None

    def update_values(self, angle_from_model):
        current_time = timeit.default_timer()
        self.prev_angle = self.angle
        self.angle = angle_from_model
        if not self.initial_state_set:
            self.initial_state_set = True
            # initial state
            self.f.x = np.array([angle_from_model, 0.])
        else:
            self.f.predict()
            self.f.update([angle_from_model])
            try:
                self.angular_velocity = (angle_from_model - self.prev_angle) / (current_time - self.calc_time)
            except:
                pdb.set_trace()

        self.calc_time = current_time

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

        self.write_bags = True

        self.AD = AngleDetector(writeImages=False, showImages=False, cv2Image=True)

        self.Model_vel_track = ModelVelocityTracker()

        self.endpoint = None

        self.finish_angle = None
        
        self.start_time = None
        self.model_started = False

        self.start_angle = None
        self.current_angle = None

        self.gripper_data = 0

        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        while not (self.gripper_data or rospy.is_shutdown()):
            rospy.sleep(1)
            print("Waiting for gripper!")

        self.gt_vel_pub = rospy.Publisher('/gt_vel', Float64, queue_size=1)
        self.model_vel_pub = rospy.Publisher('/model_vel', Float64, queue_size=1)

        self.model_kalman_vel_pub = rospy.Publisher('/model_kalman_vel', Float64, queue_size=1)
        self.model_angle_error_pub = rospy.Publisher('/model_ang_error', Float64, queue_size=1)

        self.gt_angle_pub = rospy.Publisher('/gt_ang', Float64, queue_size=1)
        self.model_angle_pub = rospy.Publisher('/model_ang', Float64, queue_size=1)

        self.use_papilarray = True
        self.digit_serials = ['D20235', 'D20226']

        self.image_topic = "/realsense/rgb"
        self.current_image = 0
        self.collect_data_flag = None
        self.collection_info = None
        # self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.collection_flag_sub = rospy.Subscriber('/collect_data', DataCollectState, self.collect_flag_callback)
        self.controller_data_sub = rospy.Subscriber('/controller_data', ControllerData, self.controller_data_callback)

        self.controller_data = None

        self.window_vec = None #torch.zeros(1, 15, 142)

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
            print("Tactile running")
            
            while self.controller_data is None:
                continue

            if self.controller_data.modelType == 'lstm':
                self.Model = RegressionLSTM("cpu", 142, 500, 3, 0.15, 0, DataMode.BOTH)
                self.Model.load_state_dict(torch.load("./models/"+str(self.controller_data.objectType)+"_lstm_finetune_model.zip")) 
                # self.Model.load_state_dict(torch.load("./random_split_lstm"))
                self.Model.eval()
                self.h_n = None
                self.c_n = None
                self.use_gt = False
            elif self.controller_data.modelType == 'mlp':
                self.Model = WindowMLP("cpu", 142, 500, 3, 0.15, 15, DataMode.BOTH) 
                self.Model.load_state_dict(torch.load("./models/"+str(self.controller_data.objectType)+"_mlp_finetune_model.zip"))
                self.Model.eval()
                self.use_gt = False
            elif self.controller_data.modelType == 'gt':
                self.Model = RegressionLSTM("cpu", 142, 500, 3, 0.15, 0, DataMode.BOTH)
                self.Model.load_state_dict(torch.load("./models/"+str(self.controller_data.objectType)+"_lstm_finetune_model.zip"))
                self.Model.eval()
                self.h_n = None
                self.c_n = None
                self.use_gt = True

            print("Received data from pipeline")

            for _ in range(100):
                inRgb = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived

            print("Node started")
            in_loop = False
            notDone = True
            move_timer = 0
            while not rospy.is_shutdown():
                
                if self.collect_data_flag: ##Start data collection

                    if self.write_bags:
                        if not in_loop:
                            bag0 = rosbag.Bag('./controller_bags_finetuned_tactile/'+str(self.controller_data.objectType)+'_'+str(self.controller_data.modelType)+'_data_'+str(int(math.floor(time.time())))+"_tactile.bag", 'w') 
                            bag1 = rosbag.Bag('./controller_bags_finetuned_results/'+str(self.controller_data.objectType)+'_'+str(self.controller_data.modelType)+'_data_'+str(int(math.floor(time.time())))+"_results.bag", 'w')
                            bag0.write('metadata', self.collection_info)
                            bag1.write('metadata', self.collection_info)
                            bag0.write('controller_data',self.controller_data)
                            bag1.write('controller_data',self.controller_data)
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

                    # Use model here
                    data = tactile_data_to_df(prev_0, prev_1)  
                    data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)
                    if self.controller_data.modelType == 'lstm' or self.controller_data.modelType == 'gt':
                        output, self.h_n, self.c_n, = self.Model(data, self.h_n, self.c_n)
                    elif self.controller_data.modelType == 'mlp':
                        if self.window_vec is None:
                            self.window_vec = torch.zeros(1, 15, 142)
                            self.window_vec[:,:] = data[:,:1]
                        self.window_vec = torch.roll(self.window_vec, 1, 1)
                        self.window_vec[:,-1,:] = data[:,0,:]
                        output = self.Model(self.window_vec).unsqueeze(0)
                    
                    # pdb.set_trace()

                    self.AD.update_angle(inRgb)

                    self.start_time = self.start_time if self.start_time is not None else rospy.Time.now()

                    if not self.model_started and (rospy.Time.now() - self.start_time).to_sec() > 0.5:
                        self.model_started = True
                        # self.command_gripper(gripper_position_msg(158))
                    elif not self.model_started:
                        print("skipping till 0.5s")
                    
                    gt_vel = self.AD.getAngularVelocity()
                    self.gt_vel_pub.publish(gt_vel)

                    # pdb.set_trace()
                    
                    pred_vel = output[:,:,0].item() * 400
                    self.model_vel_pub.publish(pred_vel)

                    pred_pos = output[:,:,1].item() * 90

                    self.Model_vel_track.update_values(pred_pos)
                    self.model_kalman_vel_pub.publish(self.Model_vel_track.angular_velocity)

                    self.model_angle_pub.publish(pred_pos)
                    gt_pos = self.AD.getAngle()
                    self.gt_angle_pub.publish(gt_pos)

                    self.model_angle_error_pub.publish(pred_pos - gt_pos)

                    if self.start_angle is None:
                        self.start_angle = gt_pos
                        self.current_angle = self.start_angle

                    if self.use_gt:
                        self.current_angle = gt_pos #+ self.AD.startAngle
                        fwd_vel = gt_vel
                    else:
                        self.current_angle = pred_pos #+ self.AD.startAngle
                        fwd_vel = pred_vel

                    # future predict the angle of the object, offset by time delay
                    time_delay = 5/60
                    fwd_angle_gt = gt_pos + gt_vel * time_delay
                    fwd_angle_model = pred_pos + pred_vel * time_delay

                    fwd_angle = fwd_angle_gt if self.use_gt else fwd_angle_model

                    # force system to use gt vel measurement
                    # fwd_vel = gt_vel
                    # fwd_angle = self.current_angle + fwd_vel * time_delay
                    
                    not_done_print = 'True' if notDone else 'False'

                    print('Not Done: %s, gt_angle: %.2f, fwd_angle: %.2f, model_angle: %.2f, model_vel: %.2f' % (not_done_print, gt_pos, fwd_angle, pred_pos, pred_vel))

                    goal = self.controller_data.targetRotationAngle
                    if abs(fwd_angle - goal) < 1 or fwd_angle > goal:
                        print("************** CLOSING *************")
                        # print('  - fwd_angle = %.2f' % (fwd_angle))
                        if self.endpoint is None:
                            self.endpoint = {'gt_angle':gt_pos, 'model_angle':pred_pos, 'fwd_angle_gt':fwd_angle_gt, 'fwd_angle_model':fwd_angle_model}
                        self.close_gripper(self.controller_data.closeWidth)
                        self.finish_angle = gt_pos
                        notDone = False
                        # self.collect_data_flag = False

                    # canMove = (current_time - self.last_move_time) > 0.05
                    move_timer += 1  # 10                              45
                    if fwd_vel < 20 and notDone and move_timer >= 45 and self.model_started:
                        print("loooooosen")
                        self.close_gripper(self.gripper_data.gPO - 1)
                        move_timer = 0

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

                        pred_vel_kal_msg = Float64()
                        pred_vel_kal_msg.data = self.Model_vel_track.angular_velocity
                        bag1.write('pred_velocity_kalman_filtered', pred_vel_kal_msg)

                        angle_error_msg = Float64()
                        angle_error_msg.data = pred_pos - gt_pos
                        bag1.write('angle_error', angle_error_msg)

                        vel_error_msg = Float64()
                        vel_error_msg.data = pred_vel - gt_vel
                        bag1.write('vel_error', vel_error_msg)

                        done_flag_msg = Bool()
                        done_flag_msg.data = notDone
                        bag1.write('not_done_flag', done_flag_msg)

                    if self.finish_angle is not None:
                        print("finish angle: ", self.finish_angle)
                elif in_loop and self.collect_data_flag is False:

                    if self.endpoint is not None:
                        print('\Stop point data:\n  - gt angle = %.2f\n  - fwd gt angle = %.2f\n  - model angle = %.2f\n  - fwd model angle = %.2f' % (self.endpoint['gt_angle'],self.endpoint['fwd_angle_gt'],self.endpoint['model_angle'],self.endpoint['fwd_angle_model']))
                    self.endpoint = None 

                    in_loop = False
                    notDone = True
                    move_timer = 0
                    ##TODO: restart everything
                    if self.write_bags:
                        if bag0 is not None:
                            bag0.close()
                        
                        if bag1 is not None:
                            bag1.close()

                    if self.controller_data.modelType == 'lstm':
                        self.Model = RegressionLSTM("cpu", 142, 500, 3, 0.15, 0, DataMode.BOTH)
                        self.Model.load_state_dict(torch.load("./models/"+str(self.controller_data.objectType)+"_lstm_finetune_model.zip"))
                        # self.Model.load_state_dict(torch.load("./random_split_lstm"))d
                        self.Model.eval()
                        self.h_n = None
                        self.c_n = None
                        self.use_gt = False
                    elif self.controller_data.modelType == 'mlp':
                        self.Model = WindowMLP("cpu", 142, 500, 3, 0.15, 15, DataMode.BOTH) 
                        self.Model.load_state_dict(torch.load("./models/"+str(self.controller_data.objectType)+"_mlp_finetune_model.zip"))
                        self.Model.eval()
                        self.use_gt = False
                    elif self.controller_data.modelType == 'gt':
                        self.Model = RegressionLSTM("cpu", 142, 500, 3, 0.15, 0, DataMode.BOTH)
                        self.Model.load_state_dict(torch.load("./models/"+str(self.controller_data.objectType)+"_lstm_finetune_model.zip"))
                        self.Model.eval()
                        self.h_n = None
                        self.c_n = None
                        self.use_gt = True

                    self.finish_angle = None
                    self.window_vec = None

                    self.start_time = None
                    self.model_started = False

                    self.start_angle = None
                    self.current_angle = None

                    self.AD = AngleDetector(writeImages=False, showImages=False, cv2Image=True)

                    self.Model_vel_track = ModelVelocityTracker()

                    print("Reset!")



    def sensor_0_callback(self, sensor_state_msg):
        self.tactile_data_0 = (self.tactile_data_0[0] + 1, sensor_state_msg)

    def sensor_1_callback(self, sensor_state_msg):
        self.tactile_data_1 = (self.tactile_data_1[0] + 1, sensor_state_msg)

    def collect_flag_callback(self, bool_msg):
        self.collection_info = bool_msg
        self.collect_data_flag = bool_msg.data

    def controller_data_callback(self, controller_data_msg):
        self.controller_data = controller_data_msg

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