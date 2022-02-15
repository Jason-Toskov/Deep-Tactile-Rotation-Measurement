#!/usr/bin/env python3

from this import d
import cv2
import depthai as dai
import rospy
import rosbag
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Header
from papillarray_ros_v2.msg import SensorState

import time, timeit
import math
from cv_bridge import CvBridge
from digit_interface import Digit

from grasp_executor.msg import DataCollectState

from enum import Enum

class DataCollectionMode(Enum):
    Contactile = 0
    Digit = 1
    Both = 2


class DataBagger:
    def __init__(self):
        rospy.init_node("Data_bagger", anonymous=True)

        # self.use_papilarray = False
        self.digit_serials = ['D20235', 'D20226']
        self.SENSORS = DataCollectionMode.Contactile

        self.image_topic = "/realsense/rgb"
        self.current_image = 0
        self.collect_data_flag = False
        self.collection_info = None
        # self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.collection_flag_sub = rospy.Subscriber('/collect_data', DataCollectState, self.collect_flag_callback)  ##TODO
        
        ##TODO: Node to read tactile data
        if self.SENSORS == DataCollectionMode.Contactile:
            self.tactile_data_0 = (0, None)
            self.tactile_data_1 = (0, None)
                    
            self.tactile_0_sub = rospy.Subscriber('/hub_0/sensor_0', SensorState, self.sensor_0_callback)
            self.tactile_1_sub = rospy.Subscriber('/hub_0/sensor_1', SensorState, self.sensor_1_callback)
        elif self.SENSORS == DataCollectionMode.Digit:
            d0 = Digit(self.digit_serials[0]) # Unique serial number
            d0.connect()
            d0.set_resolution(Digit.STREAMS["QVGA"])
            d0.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

            d1 = Digit(self.digit_serials[1]) # Unique serial number
            d1.connect()
            d1.set_resolution(Digit.STREAMS["QVGA"])
            d1.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])
        else:
            d0 = Digit('D20235') # Unique serial number
            d0.connect()
            d0.set_resolution(Digit.STREAMS["QVGA"])
            d0.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

            self.tactile_data_0 = (0, None)
            self.tactile_0_sub = rospy.Subscriber('/hub_0/sensor_0', SensorState, self.sensor_0_callback)


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


            # bag = rosbag.Bag('./data_'+str(int(math.floor(time.time())))+".bag", 'w') 
            # rospy.loginfo('bag created')
            if self.SENSORS == DataCollectionMode.Contactile:
                while self.tactile_data_0[1] == None or self.tactile_data_1[1] == None:
                    continue
            elif self.SENSORS == DataCollectionMode.Both:
                while self.tactile_data_0[1] == None:
                    continue

            rospy.loginfo("Tactile running, node started")
               
            while not rospy.is_shutdown():
               
                if self.collect_data_flag: ##Start data collection
                    rospy.loginfo("Data collection beginning")
                    bag = None
                    try:
                        ##init a bag
                        # rospy.loginfo('reached try statement')
                        bag = rosbag.Bag('./recorded_data_bags/data_'+str(int(math.floor(time.time())))+".bag", 'w') 
                        bag.write('metadata', self.collection_info)
                        
                        while self.collect_data_flag:

                            t2 = timeit.default_timer()
                            print(1 / (t2 - t1))
                            t1 = timeit.default_timer()
                            inRgb = qRgb.get().getCvFrame()  # blocking call, will wait until a new data has arrived

                            if self.SENSORS == DataCollectionMode.Digit:
                                d0frame = d0.get_frame()
                                d1frame = d1.get_frame()

                                d0msg = bridge.cv2_to_imgmsg(d0frame)
                                d1msg = bridge.cv2_to_imgmsg(d1frame)

                            elif self.SENSORS == DataCollectionMode.Both:
                                d0frame = d0.get_frame()
                                d0msg = bridge.cv2_to_imgmsg(d0frame)


                            time_now = rospy.Time.from_sec(time.time())
                            header = Header()
                            header.stamp = time_now
                            cv_image = bridge.cv2_to_imgmsg(inRgb)
                            
                            # cv2.imshow("angle", inRgb)
                            # cv2.waitKey(1)

                            if self.SENSORS == DataCollectionMode.Contactile:
                                if prev_0 == self.tactile_data_0[0]:
                                    print("uh oh repeated data")

                                if prev_1 == self.tactile_data_1[1]:
                                    print("uh oh repeated data")

                                prev_0 = self.tactile_data_0[0]
                                prev_1 = self.tactile_data_1[0]
                            elif self.SENSORS == DataCollectionMode.Both:
                                if prev_0 == self.tactile_data_0[0]:
                                    print("uh oh repeated data")

                                prev_0 = self.tactile_data_0[0]
                                

                            bag.write('time', header)
                            bag.write('image', cv_image) #Save an image

                            if self.SENSORS == DataCollectionMode.Contactile:
                                bag.write('tactile_0', self.tactile_data_0[1]) #save forces
                                bag.write('tactile_1', self.tactile_data_1[1])
                            elif self.SENSORS == DataCollectionMode.Digit:
                                bag.write('digit_0', d0msg)
                                bag.write('digit_1', d1msg)
                            else:
                                bag.write('tactile_0', self.tactile_data_0[1]) #save forces
                                bag.write('digit_0', d0msg)
                            # cv2.imshow("rgb", inRgb)

                            # cv2.waitKey(1)

                            # rate.sleep()#sleep for rate
                        ##once done, save bag
                    finally:
                        if bag is not None:
                            bag.close()
                            rospy.loginfo("Data collection ended")
                        else: 
                            rospy.loginfo("??????")

                # Retrieve 'bgr' (opencv format) frame
                #     break


    def sensor_0_callback(self, sensor_state_msg):
        self.tactile_data_0 = (self.tactile_data_0[0] + 1, sensor_state_msg)

    def sensor_1_callback(self, sensor_state_msg):
        self.tactile_data_1 = (self.tactile_data_1[0] + 1, sensor_state_msg)

    def collect_flag_callback(self, bool_msg):
        self.collection_info = bool_msg
        self.collect_data_flag = bool_msg.data
        
DataBagger()