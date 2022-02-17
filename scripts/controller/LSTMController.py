
import torch
from torch import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2, rospy, timeit
from papillarray_ros_v2.msg import SensorState
import depthai as dai
from std_msgs.msg import Float64
import matplotlib.pyplot as plt
from blob_detector import AngleDetector
from multiprocessing import Process, Queue
from RegressionLSTM import RegressionLSTM
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg

import sys
sys.path.append('/home/acrv/new_ws/src/grasp_executor/scripts')
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg, gripper_position_msg




class MeasureVelocity():

    def __init__(self):
        self.LSTM = RegressionLSTM("cpu", 142, 500, 3, 0.15)

        self.AD = AngleDetector(writeImages=False, showImages=False, cv2Image=True)

        self.LSTM.load_state_dict(torch.load("weights"))
        self.LSTM.eval()

        self.tactile_data_0 = (0, None)
        self.tactile_data_1 = (0, None)

        self.tactile_q_0 = Queue(maxsize=1)
        self.tactile_q_1 = Queue(maxsize=1)

        self.h_n = None
        self.c_n = None

        self.start_time = None
        self.lstm_started = False

        self.goal = 45
        self.last_move_time = 0

        self.current_angle = 0
        self.prev_data = None

        self.PUBLISH = True
        self.USE_GT_ANGLE = False

        self.q = Queue()
        self.width_q = Queue(maxsize=1)
        self.gripper_msg_output = Queue(maxsize=1)
        
        self.gt_ang_output = Queue()
        self.gt_vel_output = Queue()
        self.lstm_ang_output = Queue()
        self.lstm_vel_output = Queue()

        self.prev_width = None

        self.tactile_0_sub = rospy.Subscriber('/hub_0/sensor_0', SensorState, self.sensor_0_callback)
        self.tactile_1_sub = rospy.Subscriber('/hub_0/sensor_1', SensorState, self.sensor_1_callback)

        # self.gripper_data = 0

        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        while not (self.width_q.empty() or rospy.is_shutdown()):
            rospy.sleep(1)
            rospy.loginfo("Waiting for gripper!")

        self.gt_pub = rospy.Publisher('/gt_vel', Float64, queue_size=1)
        self.lstm_pub = rospy.Publisher('/lstm_vel', Float64, queue_size=1)

        self.gt_angle_pub = rospy.Publisher('/gt_ang', Float64, queue_size=1)
        self.lstm_angle_pub = rospy.Publisher('/lstm_ang', Float64, queue_size=1)

        self.p1 = Process(target=self.getCameraFrame)
        self.p2 = Process(target=self.control)

        self.p1.start()
        self.p2.start()

        while not rospy.is_shutdown():
            if not self.gripper_msg_output.empty():
                self.gripper_pub.publish(self.gripper_msg_output.get())

            if not self.lstm_ang_output.empty():
                self.lstm_angle_pub.publish(self.lstm_ang_output.get())

            if not self.lstm_vel_output.empty():
                self.lstm_pub.publish(self.lstm_vel_output.get())

            if not self.gt_ang_output.empty():
                self.gt_angle_pub.publish(self.gt_ang_output.get())

            if not self.gt_vel_output.empty():
                self.gt_pub.publish(self.gt_vel_output.get())



        # self.p2.join()

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
        if self.prev_data is None and (self.width_q.empty() or self.q.empty()):
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
            # print("loooooosen")
            self.slightly_open_gripper()

        return False

    def slightly_open_gripper(self):
        command = outputMsg.Robotiq2FGripper_robot_output()

        self.prev_width = self.width_q.get() if not self.width_q.empty() else self.prev_width

        command.rPR = self.prev_width - 1
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

        self.last_move = timeit.default_timer()

        if self.PUBLISH:
            self.gripper_msg_output.put(command)

    def gripper_state_callback(self, data):
        self.width_q.put(data.gPO)

    def close_gripper(self, width=255):
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rPR = width
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

        if self.PUBLISH:
            self.gripper_msg_output.put(command)

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

        # Linking
        prev_0 = 0
        prev_1 = 0

        camRgb.preview.link(xoutRgb.input)

        # Connect to device and start pipeline
        with dai.Device(pipeline) as device:

            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
            print("loaded device")

            t1 = timeit.default_timer()

            while self.tactile_q_0.empty() or self.tactile_q_1.empty():
                continue

            for _ in range(100):
                inRgb = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived

            print("Tactile running, node started")
               
            while not rospy.is_shutdown():
               
                inRgb = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived

                                
                if self.tactile_q_0.empty() or self.tactile_q_1.empty():
                    print("uh oh repeated data")
                
                self.tactile_data_0 = self.tactile_data_0 if self.tactile_q_0.empty() else self.tactile_q_0.get()
                self.tactile_data_1 = self.tactile_data_1 if self.tactile_q_1.empty() else self.tactile_q_1.get()

                # prev_0 = self.tactile_data_0[1]
                # prev_1 = self.tactile_data_1[1]

                data = self.tactile_data_to_df(self.tactile_data_0, self.tactile_data_1)  
                data = torch.Tensor(data).unsqueeze(0).unsqueeze(0)
                output, self.h_n, self.c_n, = self.LSTM(data, self.h_n, self.c_n)

                self.AD.update_angle(inRgb)

                self.start_time = self.start_time if self.start_time is not None else rospy.Time.now()
               
                # self.gt_pub.publish(self.AD.getAngularVelocity())
                # self.lstm_pub.publish(output*400)
                self.gt_vel_output.put(self.AD.getAngularVelocity())
                self.lstm_vel_output.put(output.item()*400)

                self.current_angle += output.item()*400*1/60
                # print(self.current_angle)

                print(f"lstm_output {(output*400).item():0.2f}, gt_output {self.AD.getAngularVelocity():0.2f}")
                print(f"integrated angle {self.current_angle:0.2f} gt angle {self.AD.getAngle()}")

                # self.lstm_angle_pub.publish(self.current_angle)
                # self.gt_angle_pub.publish(self.AD.getAngle())
                self.lstm_ang_output.put(self.current_angle)
                self.gt_ang_output.put(self.AD.getAngle())          
                if self.USE_GT_ANGLE:
                    self.q.put((timeit.default_timer(), self.AD.getAngle(), self.AD.getAngularVelocity()))
                else:
                    self.q.put((timeit.default_timer(), self.current_angle, output.item()*400))



    def sensor_0_callback(self, sensor_state_msg):
        # print("sensor 0 callback")
        # self.tactile_data_0 = (self.tactile_data_0[0] + 1, sensor_state_msg)
        self.tactile_q_0.put(sensor_state_msg)


    def sensor_1_callback(self, sensor_state_msg):
        # print("sensor 1 callback")

        # self.tactile_data_1 = (self.tactile_data_1[0] + 1, sensor_state_msg)
        self.tactile_q_1.put(sensor_state_msg)


    def command_gripper(self, grip_msg):
        self.gripper_pub.publish(grip_msg)

    def tactile_data_to_df(self,tac_0, tac_1): 
        cols_sensor = ['gfX', 'gfY', 'gfZ', 'gtX', 'gtY', 'gtZ', 'friction_est', 'target_grip_force']
        cols_pillar = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ', 'in_contact']
        l = []
        for i in range(2):
            tac = tac_0 if i==0 else tac_1

            for attr in cols_sensor:
                l.append(getattr(tac, attr))

            for j in range(9):
                for attr in cols_pillar:
                    l.append(getattr(tac.pillars[j], attr))
        return np.array(l)


if __name__ == "__main__":
    rospy.init_node("maybe")
    MeasureVelocity()
    rospy.spin()
