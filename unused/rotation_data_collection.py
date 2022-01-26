#!/usr/bin/env python
import rospy
import rosbag
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from papillarray_ros_v2.msg import SensorState

import time
import math

class DataBagger:
    def __init__(self):
        rospy.init_node("Data_bagger", anonymous=True)

        self.image_topic = "/realsense/rgb"
        self.current_image = 0
        self.collect_data_flag = False
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.collection_flag_sub = rospy.Subscriber('/collect_data', Bool, self.collect_flag_callback)  ##TODO
        
        ##TODO: Node to read tactile data
        self.tactile_data_0 = 0
        self.tactile_data_1 = 0
        self.tactile_0_sub = rospy.Subscriber('/hub_0/sensor_0', SensorState, self.sensor_0_callback)
        self.tactile_1_sub = rospy.Subscriber('/hub_0/sensor_1', SensorState, self.sensor_1_callback)


        print("Init!")

    def image_callback(self, img_msg):
        self.current_image = img_msg

    def collect_flag_callback(self, bool_msg):
        self.collect_data_flag = bool_msg.data

    def sensor_0_callback(self, sensor_state_msg):
        self.tactile_data_0 = sensor_state_msg

    def sensor_1_callback(self, sensor_state_msg):
        self.tactile_data_1 = sensor_state_msg


    def main(self):
        rate = rospy.Rate(20)

        while not ((self.current_image and self.tactile_data_0 and self.tactile_data_1) or rospy.is_shutdown()):
            rospy.sleep(1)
            rospy.loginfo("Waiting for camera/sensors!")

        rospy.loginfo("Node started!")
        while not rospy.is_shutdown():

            if self.collect_data_flag: ##Start data collection
                rospy.loginfo("Data collection beginning")
                bag = None
                try:
                    ##init a bag
                    rospy.loginfo('reached try statement')
                    bag = rosbag.Bag('./data_'+str(int(math.floor(time.time())))+".bag", 'w') 
                    rospy.loginfo('bag created')
                    while self.collect_data_flag:
                        time_now = time.time()#Get current time
                        bag.write('image', self.current_image) #Save an image
                        bag.write('tactile_0', self.tactile_data_0) #save forces
                        bag.write('tactile_1', self.tactile_data_1)
                        rate.sleep()#sleep for rate
                    ##once done, save bag
                finally:
                    if bag is not None:
                        bag.close()
                        rospy.loginfo("Data collection ended")
                    else: 
                        rospy.loginfo("??????")
            
            
            rate.sleep()


if __name__ == "__main__":
    try:
        data_class = DataBagger()
        data_class.main()
    except KeyboardInterrupt:
        pass