#!/usr/bin/env python
import rospy
import sys

import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Header, Bool
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState
from sensor_msgs.msg import JointState, PointCloud2, Image
from actionlib_msgs.msg import GoalStatusArray
from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg

from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg, gripper_position_msg
from util import dist_to_guess, vector3ToNumpy, move_ur5

class RotationMeasurer():
    def __init__(self):
        rospy.init_node("Rotation_measurement", anonymous=True)
        self.grasp_loc_joints = [0.02149745263159275, -1.8356507460223597, -1.8032754103290003, -1.0827692190753382, 1.5707544088363647, -3.5587941304981996e-05]
        self.grasp_loc_offset_joints = [0.02149745263159275, -1.723703686391012, -1.4892142454730433, -1.5087464491473597, 1.5707664489746094, -3.5587941304981996e-05]
        self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
        self.dont_display_plan = True
        self.gripper_data = 0
        self.close_width = 170
        self.slip_width = 160

        self.peturbed_joints = [0.02149745263159275, -1.723703686391012, -1.4892142454730433, -1.5087464491473597, 1.5707664489746094, -3.5587941304981996e-05]
        self.peturbed_joints[3] += 3*0.523/4

         #### Rospy startups ####

        # Moveit stuff
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # Gripper nodes
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.collection_flag = Bool()
        self.collection_flag.data = False
        self.collection_flag_pub = rospy.Publisher('/collect_data', Bool, queue_size=1)##TODO: Publisher to output when to start bagging
        self.collection_flag_pub.publish(self.collection_flag)

        print("Init!")

    def gripper_state_callback(self, data):
        self.gripper_data = data

    # Use class variables to move to a joint angle pose
    def move_to_joint_position(self, joint_array, plan=None):
        move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, joint_array, plan, no_confirm=self.dont_display_plan)

    # Publish a msg to the gripper
    def command_gripper(self, grip_msg):
        self.gripper_pub.publish(grip_msg)

    def main(self):
        rate = rospy.Rate(1)

        while not (self.gripper_data or rospy.is_shutdown()):
            rospy.sleep(1)
            rospy.loginfo("Waiting for gripper!")

        # Initialize gripper
        self.command_gripper(reset_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(activate_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(close_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(open_gripper_msg())
        rospy.sleep(.1)
        rospy.loginfo("Gripper active")

        # Go to move home position using joint definition
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.1)
        rospy.loginfo("Moved to Home Position")

        while not rospy.is_shutdown():
            rospy.loginfo("Moving above object")
            self.move_to_joint_position(self.grasp_loc_offset_joints)
            rospy.sleep(0.1)

            rospy.loginfo("Moving down to object")
            self.move_to_joint_position(self.grasp_loc_joints)
            rospy.sleep(0.1)

            # Ask user to put object into position and press enter
            in_flag = raw_input("\nPlease put object into gripper fingers and press 'y': ")
            while in_flag is not 'y' and not rospy.is_shutdown():
                in_flag = raw_input("Input was not 'y': ")

            self.command_gripper(gripper_position_msg(self.close_width))
            rospy.sleep(.5)

            rospy.loginfo("Moving up")
            self.move_to_joint_position(self.grasp_loc_offset_joints)
            rospy.sleep(0.1)

            rospy.loginfo("Peturbing end effector")
            #Can add an EE angle pertubation here
            self.move_to_joint_position(self.peturbed_joints)

            raw_input("Press enter to start data collection: ")

            # Send a request to a service to bag some data (maybe by publishing to some topic)
            self.collection_flag.data = True
            self.collection_flag_pub.publish(self.collection_flag)

            #Loosen gripper
            self.command_gripper(gripper_position_msg(self.slip_width))

            # Pause and wait for user to say rotation is done
            raw_input("Press enter when object is fully rotated: ")

            # End data collection (by publishing a stop to the topic)
            self.collection_flag.data = False
            self.collection_flag_pub.publish(self.collection_flag)
            rospy.sleep(1)

            # Drop object completely
            self.command_gripper(open_gripper_msg())

            #Wait for user input saying theyve removed the object
            raw_input("Press enter when object is removed from workspace: ")

            rate.sleep()


if __name__ == "__main__":
    try:
        data_class = RotationMeasurer()
        data_class.main()
    except KeyboardInterrupt:
        pass