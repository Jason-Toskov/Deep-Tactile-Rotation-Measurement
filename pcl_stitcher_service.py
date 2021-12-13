#!/usr/bin/env python

import roslib; roslib.load_manifest('laser_assembler')
import rospy; from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import tf
from tf import TransformListener
from geometry_msgs.msg import PoseStamped
import moveit_commander
import moveit_msgs.msg
from grasp_executor.srv import PCLStitch
from util import move_ur5

import pdb

SCAN_JOINTS = [[

],[

]]

class PCLStitcher:
    def __init__(self):
        rospy.init_node("generate_pcl_service")
        rospy.wait_for_service("assemble_scans2")

        self.pcl_rosmsg = 0

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.PCL_publisher = rospy.Publisher("/my_cloud_in", PointCloud2, queue_size=1)
        self.PCL_reader = rospy.Subscriber("/realsense/cloud", PointCloud2, self.cloud_callback)

        self.assemble_scans = rospy.ServiceProxy('assemble_scans2', AssembleScans2)

        self.PCL_server = rospy.Service('generate_pcl', PCLStitch, self.generate_pcl)

    def cloud_callback(self, pcl):
        self.pcl_rosmsg = pcl
        self.pcl_rosmsg.header.stamp.secs = rospy.Time.now().secs
        self.pcl_rosmsg.header.stamp.nsecs = rospy.Time.now().nsecs

    def generate_pcl(self, req):
        ## TODO: Generate and return concatenated pcl here
        rospy.loginfo("Reached PCL service")

        time_start = rospy.Time.now()

        for joints in SCAN_JOINTS(req.mode):
            move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, joints)
            rospy.sleep(1)
            self.PCL_publisher.publish(self.pcl_rosmsg)

        resp = self.assemble_scans(time_start, rospy.Time.now())

        return resp


if __name__ == "__main__":
    try:
        PCLStitcher()
        rospy.spin()
    except KeyboardInterrupt:
        pass
