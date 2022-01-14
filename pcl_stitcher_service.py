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
from grasp_2_boxes import State

import pdb

# SCAN_JOINTS = [[
# # ### Potential view joint angles
# # self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
# [0.24985386431217194, -0.702608887349264, -2.0076406637774866, -1.7586587111102503, 1.5221580266952515, 0.25777095556259155],
# # [0.09033633768558502, -2.460919205342428, -0.5586937109576624, -2.818702522908346, 1.687928318977356, 0.03240497037768364],

# # #Middle, retracted to extended
# # [0.18934065103530884, -1.4652579466449183, -1.4612248579608362, -2.124643627797262, 1.6266496181488037, 0.18911968171596527],
# [0.16507820785045624, -1.728558365498678, -1.2855127493487757, -2.0006192366229456, 1.6261470317840576, 0.18910770118236542],
# # [0.10739026963710785, -1.9473517576800745, -0.9374383131610315, -2.05618125597109, 1.6247107982635498, 0.1306973397731781],
# [0.1277136355638504, -2.372087303792135, -0.5449422041522425, -2.5323551336871546, 1.6263383626937866, 0.13070932030677795],
# [0.1277376115322113, -2.37207538286318, -0.544870678578512, -2.8464210669146937, 1.626314401626587, 0.1307213008403778],

# [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272],
# # #Side far, retracted to extended
# [-0.7505858580218714, -1.3659656683551233, -1.6535900274859827, -2.056385342274801, 2.3304872512817383, -0.48804933229555303],
# [-0.38263065019716436, -2.0775330702411097, -0.9685853163348597, -2.589614216481344, 2.33158802986145, -0.48804933229555303],
# [-0.14996081987489873, -2.202012364064352, -0.6998351255999964, -2.7694557348834437, 2.078448534011841, 0.08022802323102951],


# [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272],
# # #Side close, retracted to extended
# [0.8964406251907349, -1.3053887526141565, -1.8282573858844202, -2.20429498354067, 0.5266827344894409, 0.08080288767814636],
# [0.5870899558067322, -1.84922963777651, -1.228541676198141, -2.292896095906393, 1.212148904800415, 0.9514194130897522],
# [0.2799836993217468, -2.4212587515460413, -0.45788795152773076, -2.8320158163653772, 1.6029754877090454, 0.950688362121582],

# [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272],

# ],[

# ]]

SCAN_JOINTS = {
    State.RIGHT_TO_LEFT: [
        [0.12033622711896896, -1.4533870855914515, -1.5536163488971155, -1.747192684804098, 1.416150450706482, 1.6476819515228271],
        [-0.7302053610431116, -1.9902375380145472, -0.9067710081683558, -2.4823296705829065, 2.159045696258545, 0.28434035181999207],
        [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272],
    ],
    State.LEFT_TO_RIGHT: [
        
        [0.5746720433235168, -1.4123638311969202, -1.7548344771014612, -1.3990066687213343, 1.6094144582748413, -1.0251277128802698],
        [0.8106227517127991, -1.684913460408346, -1.4413684050189417, -2.00739032426943, 1.2154176235198975, -0.5774405638324183],
        [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272],
    ]
}


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
        rospy.loginfo("Reached PCL service")

        time_start = rospy.Time.now()

        for joints in SCAN_JOINTS[State(req.mode)]:
            move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, joints, no_confirm=True)
            rospy.sleep(1)
            self.PCL_publisher.publish(self.pcl_rosmsg)

        resp = self.assemble_scans(time_start, rospy.Time.now())

        return resp.cloud


if __name__ == "__main__":
    try:
        PCLStitcher()
        rospy.spin()
    except KeyboardInterrupt:
        pass
