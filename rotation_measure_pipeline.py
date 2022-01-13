#!/usr/bin/env python
import rospy

import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Header
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState
from sensor_msgs.msg import JointState, PointCloud2, Image
from actionlib_msgs.msg import GoalStatusArray
from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg

from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from util import dist_to_guess, vector3ToNumpy, move_ur5

class RotationMeasurer():
    def __init__(self):
        self.grasp_loc_joints = []
        self.grasp_loc_offset_joints = []
        self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
        self.dont_display_plan = False
        self.gripper_data = 0

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

        ##TODO: Publisher to output when to start bagging

        rospy.init_node("Rotation_measurement", anonymous=True)
        print("Init!")

    def gripper_state_callback(self, data):
        self.gripper_data = data

    # Use class variables to move to a joint angle pose
    def move_to_joint_position(self, joint_array, plan=None):
        move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, joint_array, plan, no_confirm=self.dont_display_plan)

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

            #Can add an EE angle pertubation here
            #
            #

            # Ask user to put object into position and press enter
            in_flag = raw_input("\nPlease put object into gripper fingers and press 'y'")
            while in_flag is not 'y' and not rospy.is_shutdown():
                in_flag = raw_input("Input was not 'y': ")

            self.command_gripper(close_gripper_msg())
            rospy.sleep(.5)

            rospy.loginfo("Moving up")
            self.move_to_joint_position(self.grasp_loc_offset_joints)
            rospy.sleep(0.1)

            # Loosen gripper

            # Send a request to a service to bag some data (maybe by publishing to some topic)

            # Pause and wait for user to say rotation is done

            # End data collection (by publishing a stop to the topic)

            # Drop object completely

            #Wait for user input saying theyve removed the object

            rate.sleep()


if __name__ == "__main__":
    try:
        data_class = RotationMeasurer()
        data_class.main()
    except KeyboardInterrupt:
        pass