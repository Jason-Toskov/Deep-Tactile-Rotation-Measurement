#!/usr/bin/env python

import rospy
# import pcl
import numpy as np
# import ctypes
# import struct
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
# from random import randint

import ros_numpy.point_cloud2 as rpc2
import pdb

import tf
from tf import TransformListener

from geometry_msgs.msg import PoseStamped




class PCL_Processing:
    def __init__(self):
        rospy.init_node("PCL_processing", anonymous=True)

        self.pcl_rosmsg = 0

        self.tf_listener_ = TransformListener()

        self.tf_listener_.waitForTransform("/camera_link", "/base_link", rospy.Time(), rospy.Duration(4))

        self.pose = PoseStamped()
        self.pose.header.frame_id = 'camera_link'
        self.transform_matrix = self.tf_listener_.asMatrix("/base_link", self.pose.header)

        self.PCL_publisher = rospy.Publisher("/processed_PCL2", PointCloud2, queue_size=1)
        self.PCL_reader = rospy.Subscriber("/realsense/cloud", PointCloud2, self.cloud_callback)




    def cloud_callback(self, pcl):
        self.pcl_rosmsg = pcl

    def transformed_vector(self, x):
        x_transformed = np.matmul(self.transform_matrix, np.array([x[0], x[1], x[2], 1]))
        #pdb.set_trace()
        return x_transformed[:3]

    def check_box_bounds(self, x):
        x = self.transformed_vector(x)
        
        # if -0.78 < x[0] < -0.41 :
        #     if -0.12 < x[1] < 0.158:

        # RHS box
        if -0.610 < x[0] < -0.335 :
            if 0.140 < x[1] < 0.505:

        # # LHS box
        # if -0.580 < x[0] < -0.305 :
        #     if -0.520 < x[1] < -0.160:

                return True
        
        return False

    def main(self):
        rate = rospy.Rate(1)

        while not rospy.is_shutdown():
            self.transform_matrix = self.tf_listener_.asMatrix("/base_link", self.pose.header)
            if self.pcl_rosmsg:

                np_array = rpc2.pointcloud2_to_array(self.pcl_rosmsg)

                mask = np.array([self.check_box_bounds(x) for x in np_array])

                filtered_np_array = np_array[mask]

                new_pcl2 = rpc2.array_to_pointcloud2(filtered_np_array)
                new_pcl2.header.frame_id = 'camera_link'
                self.PCL_publisher.publish(new_pcl2)
            
            rospy.loginfo("cycle")
            rate.sleep()

if __name__ == "__main__":
    try:
        node = PCL_Processing()
        node.main()

    except KeyboardInterrupt:
        pass
    