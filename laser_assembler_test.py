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


class PCL2Processor:
    def __init__(self):
        rospy.init_node("pcl2_assembler")
        rospy.wait_for_service("assemble_scans2")

        self.pcl_rosmsg = 0

        # self.tf_listener_ = TransformListener()
        # self.tf_listener_.waitForTransform("/camera_link", "/base_link", rospy.Time(), rospy.Duration(4))

        self.PCL_publisher = rospy.Publisher("/my_cloud_in", PointCloud2, queue_size=1)
        self.PCL_reader = rospy.Subscriber("/realsense/cloud", PointCloud2, self.cloud_callback)

        self.PCL_stitched_publisher = rospy.Publisher("/processed_PCL2_stitched", PointCloud2, queue_size=1)

        self.assemble_scans = rospy.ServiceProxy('assemble_scans2', AssembleScans2)

    def cloud_callback(self, pcl):
        self.pcl_rosmsg = pcl
        self.pcl_rosmsg.header.stamp.secs = rospy.Time.now().secs
        self.pcl_rosmsg.header.stamp.nsecs = rospy.Time.now().nsecs

    def run(self):
        rate = rospy.Rate(1)

        time_to_take = rospy.Time.now()

        rospy.sleep(1)

        while not rospy.is_shutdown():

            if self.pcl_rosmsg:
                self.PCL_publisher.publish(self.pcl_rosmsg)
                last_pcl = self.pcl_rosmsg
                self.pcl_rosmsg = 0

                flag = raw_input("Do you wish to assemble the cloud [y/n]: ")

                if flag == 'y':
                    rospy.sleep(1)
                    resp = self.assemble_scans(time_to_take, rospy.Time.now())
                    print("From ",time_to_take.secs, " to ", rospy.Time.now().secs)
                    print("Last pcl timestamp = ", last_pcl.header)
                    in_interval =  time_to_take.secs < last_pcl.header.stamp.secs < rospy.Time.now().secs
                    print("pcl is in interval: ", in_interval)
                    self.PCL_stitched_publisher.publish(resp.cloud)
                    time_to_take = rospy.Time.now()
                    pdb.set_trace()
            
            rate.sleep()

    


if __name__ == "__main__":
    try: 
        pcl2_assembler = PCL2Processor()
        pcl2_assembler.run()
    except KeyboardInterrupt:
        pass