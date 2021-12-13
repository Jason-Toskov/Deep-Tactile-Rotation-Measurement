#!/usr/bin/env python

import roslib; roslib.load_manifest('laser_assembler')
import rospy; from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import tf
from tf import TransformListener
from geometry_msgs.msg import PoseStamped

import pdb



class PCLStitcher:
    def __init__(self):
        rospy.init_node("generate_pcl_service")
        rospy.wait_for_service("assemble_scans2")

        self.pcl_rosmsg = 0

        self.PCL_publisher = rospy.Publisher("/my_cloud_in", PointCloud2, queue_size=1)
        self.PCL_reader = rospy.Subscriber("/realsense/cloud", PointCloud2, self.cloud_callback)

        # self.PCL_stitched_publisher = rospy.Publisher("/processed_PCL2_stitched", PointCloud2, queue_size=1)

        self.assemble_scans = rospy.ServiceProxy('assemble_scans2', AssembleScans2)

        self.PCL_server = rospy.Service('generate_pcl', PointCloud2, self.generate_pcl)

    def cloud_callback(self, pcl):
        self.pcl_rosmsg = pcl
        self.pcl_rosmsg.header.stamp.secs = rospy.Time.now().secs
        self.pcl_rosmsg.header.stamp.nsecs = rospy.Time.now().nsecs

    def generate_pcl(self, req):
        ## TODO: Generate and return concatenated pcl here
        rospy.loginfo("Reached PCL service")
        return self.pcl_rosmsg


if __name__ == "__main__":
    try:
        PCLStitcher()
        rospy.spin()
    except KeyboardInterrupt:
        pass
