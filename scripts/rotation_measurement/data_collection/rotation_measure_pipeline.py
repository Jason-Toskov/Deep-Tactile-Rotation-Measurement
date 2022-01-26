#!/usr/bin/env python
import rospy
import sys
import numpy as np
import copy
from pyquaternion import Quaternion
import tf
import geometry_msgs
import pdb

import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Header, Bool
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState
from sensor_msgs.msg import JointState, PointCloud2, Image
from actionlib_msgs.msg import GoalStatusArray
from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg

from grasp_executor.msg import DataCollectState
from scripts.gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg, gripper_position_msg
from scripts.util import dist_to_guess, vector3ToNumpy, move_ur5

PI = np.pi

class RotationMeasurer():
    def __init__(self):
        rospy.init_node("Rotation_measurement", anonymous=True)
        self.grasp_loc_joints = [0.02149745263159275, -1.8356507460223597, -1.8032754103290003, -1.0827692190753382, 1.5707544088363647, -3.5587941304981996e-05]
        # self.grasp_loc_offset_joints = [0.02149745263159275, -1.723703686391012, -1.4892142454730433, -1.5087464491473597, 1.5707664489746094, -3.5587941304981996e-05]
        self.grasp_loc_offset_joints = [0.034072648733854294, -1.2315533796893519, -1.9322221914874476, -1.5579307715045374, 1.5709701776504517, 0.012650847434997559]



        self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
        self.dont_display_plan = True
        self.gripper_data = 0

        # self.grasp_loc_pose = self.set_pose("base_link",-0.5506,  0.0973, 0.0700,  0.4969, 0.5030, -0.4922, 0.5076)
        self.grasp_loc_pose = self.set_pose("base_link", -0.4, 0.0973, 0.07, 0.4969, 0.5030, -0.4922, 0.5076)

        self.grasp_loc_offset_pose = copy.deepcopy(self.grasp_loc_pose)
        self.offset_z = 0.2
        self.grasp_loc_offset_pose.pose.position.z += self.offset_z


        self.in_air_pertubation_angles_coeff = [0, PI/12, PI/6, PI/4, PI/3, -PI/12, -PI/6, -PI/4]
        self.on_ground_pertubation_angles_coeff = [0, PI/12, PI/6, PI/4, PI/3, -PI/12, -PI/6]
        # self.on_ground_pertubation_angles_coeff = [PI/3]

        # for the box + tape
        # self.close_width = 175
        # self.slip_width = 163

        # for the deoderant
        # self.close_width = 223
        # self.slip_width = 90

        ### Gripped width code

        self.close_width = 175###

        self.loosest_grasp = 160
        self.tightest_grasp = 165
        self.num_step = 2

        self.skip_count = 174
        self.collect_data = True

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

        self.collection_flag = DataCollectState()
        self.collection_flag.data = False
        self.collect_data_pub = rospy.Publisher('/collect_data', DataCollectState, queue_size=1)
        self.collect_data_pub.publish(self.collection_flag)

        print("Init!")

    def gripper_state_callback(self, data):
        self.gripper_data = data

    # Use class variables to move to a pose
    def move_to_position(self, grasp_pose, plan=None):
        move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, grasp_pose, plan, no_confirm=self.dont_display_plan)

    # Use class variables to move to a joint angle pose
    def move_to_joint_position(self, joint_array, plan=None):
        move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, joint_array, plan, no_confirm=self.dont_display_plan)

    # Publish a msg to the gripper
    def command_gripper(self, grip_msg):
        self.gripper_pub.publish(grip_msg)

    def set_pose(self,frame, posx, posy, posz, orx, ory, orz, orw):
        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x = posx
        pose.pose.position.y = posy
        pose.pose.position.z = posz

        pose.pose.orientation.x = orx
        pose.pose.orientation.y = ory
        pose.pose.orientation.z = orz
        pose.pose.orientation.w = orw

        return pose

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
        count = 0
        while not rospy.is_shutdown():
            for wrist_orientation in [0, PI]: ## TODO: check if pi should be positive or negative
                offset_joints = copy.deepcopy(self.grasp_loc_offset_joints)
                offset_joints[5] += wrist_orientation ##TODO get the correct element here, and then check that the rotations are right

                for width in np.round(np.linspace(self.loosest_grasp, self.tightest_grasp+1, num=self.num_step)).astype(np.uint8):
                    for ground_angle_peturb in self.on_ground_pertubation_angles_coeff:
                        for air_angle_peturb in self.in_air_pertubation_angles_coeff:

                            if count < self.skip_count:
                                count += 1
                                continue

                            rospy.loginfo("Moving above object")
                            self.move_to_joint_position(offset_joints)
                            rospy.sleep(0.5)

                            # pdb.set_trace()


                            curr_rpy = self.move_group.get_current_rpy()
                            curr_rpy[1] += ground_angle_peturb #Change the pitch (y axis)
                            
                            # print(curr_rpy)

                            currPose = copy.deepcopy(self.grasp_loc_pose) #self.move_group.get_current_pose()
                            quaternion = tf.transformations.quaternion_from_euler(curr_rpy[0], curr_rpy[1], curr_rpy[2])
                            currPose.pose.orientation.x = quaternion[0]
                            currPose.pose.orientation.y = quaternion[1]
                            currPose.pose.orientation.z = quaternion[2]
                            currPose.pose.orientation.w = quaternion[3]

                            # self.move_group.se

                            # pdb.set_trace()


                            # Rotate gripper to be close to ground here
                            # Set the pose to have an offset angle
                            rospy.loginfo("Moving down to object")
                            self.move_to_position(currPose)
                            rospy.sleep(0.1)

                            

                            # rospy.loginfo("Peturbing angle")
                            # self.move_to_position(currPose)
                            # rospy.sleep(0.1)

                            # Ask user to put object into position and press enter
                            raw_input("\nPlease put object into gripper fingers and press ENTER: ")

                            self.command_gripper(gripper_position_msg(self.close_width))
                            rospy.sleep(.5)

                            rospy.loginfo("Moving up")

                            currPose.pose.position.z += self.offset_z

                            self.move_to_position(currPose)
                            rospy.sleep(0.1)

                            rospy.loginfo("Peturbing end effector")

                            curr_rpy[1] -= ground_angle_peturb
                            curr_rpy[1] += air_angle_peturb
                            # curr_rpy_2 = self.move_group.get_current_rpy()
                            # curr_rpy_2[1] += -ground_angle_peturb#air_angle_peturb - ground_angle_peturb  #Change the pitch (y axis)

                            perturbedPose = self.move_group.get_current_pose()
                            quaternion_2 = tf.transformations.quaternion_from_euler(curr_rpy[0], curr_rpy[1], curr_rpy[2])

                            perturbedPose.pose.orientation.x = quaternion_2[0]
                            perturbedPose.pose.orientation.y = quaternion_2[1]
                            perturbedPose.pose.orientation.z = quaternion_2[2]
                            perturbedPose.pose.orientation.w = quaternion_2[3]

                            #Can add an EE angle pertubation here
                            # self.peturbed_joints = copy.copy(self.grasp_loc_offset_joints)
                            # self.peturbed_joints[3] += air_angle_peturb
                            self.move_to_position(perturbedPose)

                            raw_input("Press enter to start data collection: ")

                            # Send a request to a service to bag some data (maybe by publishing to some topic)
                            collection_info = DataCollectState()
                            collection_info.data = self.collect_data
                            collection_info.gripperTwist = round(wrist_orientation/PI*180)
                            collection_info.eeGroundRot = round(ground_angle_peturb/PI*180)
                            collection_info.eeAirRot = round(air_angle_peturb/PI*180)
                            collection_info.gripperWidth = width
                            self.collect_data_pub.publish(collection_info)

                            #Loosen gripper
                            self.command_gripper(gripper_position_msg(width))

                            # Pause and wait for user to say rotation is done
                            raw_input("Press enter when object is fully rotated: ")

                            # End data collection (by publishing a stop to the topic)
                            collection_info.data = False
                            self.collect_data_pub.publish(collection_info)
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