#!/usr/bin/env python
import rospy
import roslaunch
import tf
from tf import TransformListener

import sys
import copy 
import random
import pdb
from enum import Enum, IntEnum
from time import sleep
import numpy as np
from pyquaternion import Quaternion
import rosbag
import math
import time, timeit
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from tf import TransformListener

import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Header
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState
from sensor_msgs.msg import JointState, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from actionlib_msgs.msg import GoalStatusArray
from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg

import sys
sys.path.append('/home/acrv/new_ws/src/grasp_executor/scripts')
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from util import dist_to_guess, vector3ToNumpy, move_ur5, floatToMsg, floatArrayToMsg, intToMsg
from grasp_executor.srv import PCLStitch


import roslib; roslib.load_manifest('laser_assembler')
import rospy; from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud2, CameraInfo
from std_msgs.msg import Header
import numpy as np
import tf
from tf import TransformListener
from geometry_msgs.msg import PoseStamped
import moveit_commander
import moveit_msgs.msg
from grasp_executor.srv import PCLStitch

import sys
sys.path.append('/home/acrv/new_ws/src/grasp_executor/scripts')
from util import move_ur5

sys.path.append('/home/acrv/new_ws/src/grasp_executor/scripts/grasping_demo')

import pdb

import open3d as o3d

from ctypes import * # convert float to uint32

import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

class State(IntEnum):
    BOOTUP=0
    LEFT_TO_RIGHT=1
    RIGHT_TO_LEFT=2

STATE_TRANSITION = {
    State.LEFT_TO_RIGHT: State.RIGHT_TO_LEFT,
    State.RIGHT_TO_LEFT: State.LEFT_TO_RIGHT
}

class AgileState(Enum):
    RESET = 0
    WAIT_FOR_ONE = 1
    READY = 2

AGILE_STATE_TRANSITION = {
    AgileState.RESET: AgileState.WAIT_FOR_ONE,
    AgileState.WAIT_FOR_ONE: AgileState.READY,
    AgileState.READY: AgileState.READY
}

class GraspExecutor:

    def __init__(self):
        # Need pcl_stitcher_service.py running
        rospy.init_node('grasp_executor', anonymous=True)
        rospy.loginfo("Waiting for PCL stitching node")
        rospy.wait_for_service('generate_pcl')
        rospy.wait_for_service("assemble_scans2")
        rospy.loginfo("Node active!")

        #### Useful variables ####
        #Positions
        self.move_home_joints = [0.008503181859850883, -1.5707362333880823, -1.409212891255514, -1.7324321905719202, 1.5708622932434082, 0.008457492105662823]

        self.stable_test_joints = [0.008503181859850883, -1.5707362333880823, -1.409212891255514, -0.1542146841632288, 1.5708622932434082, 0.008457492105662823]

        # Multi-view points
        self.pose_1 = [-1.1746083394825746e-05, -0.631601635609762, -1.82951528230776, -2.1099846998797815, 1.569976568222046, -4.7985707418263246e-05]
        self.pose_2 = [0.0, -2.110682789479391, -0.23039132753481084, -2.8553083578692835, 1.5699286460876465, -3.5587941304981996e-05]
        self.pose_3 = [-0.5584028402911585, -2.0729802290545862, -0.40200978914369756, -2.636850659047262, 2.227617025375366, 0.8789638876914978]
        self.pose_4 = [0.7862164974212646, -2.1710355917560022, -0.403877083455221, -2.9715493361102503, 1.0643872022628784, -0.5000680128680628]
        # self.view_joints = [self.pose_1, self.pose_2, self.pose_3, self.pose_4]
        self.view_joints = [self.pose_2, self.pose_3, self.pose_4]

        self.drop_joints = {
            State.RIGHT_TO_LEFT: [0.8464177250862122, -1.7617242972003382, -1.3163345495807093, -1.664525334035055, 1.5956381559371948, 0.03218962997198105],
            State.LEFT_TO_RIGHT: [-0.4250834623919886, -1.76178485551943, -1.3162863890277308, -1.6644414106952112, 1.5955902338027954, 0.03218962997198105],
            }
        self.drop_joints_no_box = {
            State.RIGHT_TO_LEFT: [0.8463821411132812, -1.8050292173968714, -1.6474922339068812, -1.2900922934161585, 1.5956381559371948, 0.03232145681977272],
            State.LEFT_TO_RIGHT: [-0.42509586015810186, -1.805472199116842, -1.648043457661764, -1.2890384832965296, 1.595733642578125, 0.03230947256088257],
            }

        # Create workspace
        # (a box with the center at the origin; [minX, maxX, minY, maxY, minZ, maxZ]) 
        # X: 310, Y: -320
        # X: 640, Y: -320
        # X: 640, Y: 135
        # X: 310, Y: 135
        self.workspaces = {
            State.RIGHT_TO_LEFT: [-0.568, -0.327, 0.174, 0.501, 0, 1],
            State.LEFT_TO_RIGHT: [-0.553, -0.327, -0.511, -0.179, 0, 1],
            }

        self.workspace = [-0.640, -0.310, -0.135, 0.320, -1, 1]
        # self.workspace = [-1, 1, -1, 1, 0, 1]

        # Home robot state
        self.move_home_robot_state = self.get_robot_state(self.move_home_joints)

        # Stable test robot state
        self.stable_test_robot_state = self.get_robot_state(self.stable_test_joints)

        # Boolean parameters
        self.dont_display_plan = True
        self.choose_random = True
        self.no_boxes = True

        # Initializations
        self.state = State.BOOTUP
        self.agile_state = AgileState.WAIT_FOR_ONE

        # Initialise data values
        self.gripper_data = 0
        self.agile_data = 0

        #### Rospy startups ####

        # Moveit stuff
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # Publisher for grasp arrows
        self.pose_publisher = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)
        
        # Gripper nodes
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        # Nodes for stitched point cloud
        self.generate_pcl = rospy.ServiceProxy('generate_pcl', PCLStitch)
        self.PCL_stitched_publisher = rospy.Publisher("/processed_PCL2_stitched", PointCloud2, queue_size=1)

        # Point cloud assembler
        self.assemble_scans = rospy.ServiceProxy('assemble_scans2', AssembleScans2)

        # Point cloud
        self.PCL_publisher = rospy.Publisher("/my_cloud_in", PointCloud2, queue_size=1)
        self.PCL_reader = rospy.Subscriber("/realsense/cloud", PointCloud2, self.cloud_callback)

        # Point cloud test
        self.test_publisher = rospy.Publisher("/point_cloud_test", PointCloud2, queue_size=1)

        # Camera Info
        self.cam_info_reader = rospy.Subscriber("/realsense/camera_info", CameraInfo, self.cam_info_callback)

        # TF Listener
        self.tf_listener_ = TransformListener()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(12))
        self.tl = tf2_ros.TransformListener(self.tf_buffer)

        # Agile grasp node
        rospy.Subscriber("/detect_grasps/grasps", GraspListMsg, self.agile_callback)

        # Threshold quality value
        self.success_ratio = 0.5

        # RGB Image
        self.rgb_sub = rospy.Subscriber('/realsense/rgb', Image, self.rgb_callback)
        self.cv_image = []
        self.image_number = 0

        # Depth Image
        self.depth_sub = rospy.Subscriber('/realsense/depth', Image, self.depth_image_callback)
        self.depth_image = []

        # CV Bridge
        self.bridge = CvBridge()
        self.cv2Image = cv2Image = False

        # Data collection 
        self.collect_data = True
        # collect_data_flag = raw_input("Collect data? (y or n): ")
        # if collect_data_flag == "y":
        #     rospy.loginfo("Collecting data")
        #     self.collect_data = True
        # else:
        #     self.collect_data = False
        
        # Object ID
        self.object_id = int(raw_input("Enter object ID: "))

        # Resume 
        self.attempt_counter = 0.0
        self.success_counter = 0.0

        resume_data_flag = raw_input("New object? (y or n): ")
        if resume_data_flag == "n":
            self.attempt_counter = float(raw_input("Number of attempts?: "))
            self.success_counter = float(raw_input("Number of successes?: "))
        else:
            rospy.loginfo("Beginning new object")
        
        # Create bag
        if self.collect_data:
            self.bag = rosbag.Bag('/home/acrv/new_ws/src/grasp_executor/scripts/grasping_demo/grasp_data_bags/data_' + str(int(math.floor(time.time()))) + ".bag", 'w')

    def cam_info_callback(self, cam_info):
        self.cam_info = cam_info
    
    def cloud_callback(self, pcl):
        self.pcl_rosmsg = pcl
        self.pcl_rosmsg.header.stamp.secs = rospy.Time.now().secs
        self.pcl_rosmsg.header.stamp.nsecs = rospy.Time.now().nsecs

    def rgb_callback(self, image):
        self.rgb_image = image
        self.cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        self.image_number += 1

    def depth_image_callback(self, image):
        self.depth_image = image
    
    def agile_callback(self, data):
        self.agile_data = data
        self.agile_state = AGILE_STATE_TRANSITION[self.agile_state]

    def find_best_grasp(self, data, choose_random=True):
        timeout_time = 15 # in seconds
        robot_dist = 0.168
        offset_dist = 0.1
        max_angle = 100
        final_grasp_pose = 0
        final_grasp_pose_offset = 0
        robot_pose = 0

        num_bad_angle = 0
        num_bad_plan = 0

        poses = []
        rospy.loginfo("Success Ratio: %f", self.success_ratio)

        if choose_random:
            if self.success_ratio > 0.5:
                # Worst to best
                rospy.loginfo("Looking for BAD grasp")
                data.grasps.sort(key=lambda x : x.score, reverse=False)
                # data.grasps = data.grasps[0:100]
                noise = (self.success_ratio - 0.5) * 0.05
            else:
                # Best to worst
                rospy.loginfo("Looking for GOOD grasp")
                data.grasps.sort(key=lambda x : x.score, reverse=True)
                # data.grasps = data.grasps[0:100]
                noise = 0.0
            #Shuffle grasps
            # random.shuffle(data.grasps)
        else:
            # Sort grasps by quality
            if self.success_ratio > 0.5:
                data.grasps.sort(key=lambda x : x.score, reverse=False)
            else:
                data.grasps.sort(key=lambda x : x.score, reverse=True)

        # Back up random list
        rand_grasps = copy.deepcopy(data.grasps)
        random.shuffle(rand_grasps)

        loop_start_time = rospy.Time.now()
        # loop through grasps from high to low quality
        for g_normal, g_rand in zip(data.grasps, rand_grasps):
            if rospy.is_shutdown():
                break

            # Take from whatever the option was (sorted/random) if not timed out
            time_taken = (rospy.Time.now() - loop_start_time).secs
            if time_taken < timeout_time:
                g = g_normal
                # rospy.loginfo("Using selected shuffle")
            elif time_taken < 3*timeout_time:
                g = g_rand
                # rospy.loginfo("Using random shuffle")
            else:
                rospy.loginfo("Max search time exceeded!")
                break
            
            # Get grasp pose from agile grasp outputs
            R = np.zeros((3,3))
            R[:, 0] = vector3ToNumpy(g.approach)
            R[:, 1] = vector3ToNumpy(g.axis)
            R[:, 2] = np.cross(vector3ToNumpy(g.approach), vector3ToNumpy(g.axis))

            q = Quaternion(matrix=R)
            position =  g.surface
            # rospy.loginfo("Grasp cam orientation found!")

            #Create poses for grasp and pulled back (offset) grasp
            p_base = PoseStamped()

            p_base.pose.position.x = position.x + noise
            p_base.pose.position.y = position.y + noise
            p_base.pose.position.z = position.z + noise

            p_base.pose.orientation.x = q[1]
            p_base.pose.orientation.y = q[2]
            p_base.pose.orientation.z = q[3]
            p_base.pose.orientation.w = q[0]

            p_base_offset = copy.deepcopy(p_base)
            p_base_offset.pose.position.x -= g.approach.x * offset_dist
            p_base_offset.pose.position.y -= g.approach.y * offset_dist
            p_base_offset.pose.position.z -= g.approach.z * offset_dist

            p_base_robot = copy.deepcopy(p_base)
            p_base_robot.pose.position.x -= g.approach.x * robot_dist
            p_base_robot.pose.position.y -= g.approach.y * robot_dist
            p_base_robot.pose.position.z -= g.approach.z * robot_dist

            # Here we need to define the frame the pose is in for moveit
            p_base.header.frame_id = "base_link"
            p_base_offset.header.frame_id = "base_link"
            p_base_robot.header.frame_id = "base_link"

            # Used for visualization
            poses.append(copy.deepcopy(p_base.pose))

            # Find angle between -z axis and approach
            approach_base = np.array([g.approach.x, g.approach.y, g.approach.z])
            approach_base = approach_base / np.linalg.norm(approach_base)
            theta_approach = np.arccos(np.dot(approach_base, np.array([0,0,-1])))*180/np.pi

            # rospy.loginfo("Grasp base orientation found")  

            # If approach points up, no good            
            if theta_approach < max_angle:
                
                # Check if plan to grasp is valid
                self.move_group.set_start_state(self.move_home_robot_state)
                # self.move_group.set_pose_target(p_base)
                # plan_to_final = self.move_group.plan()

                # TODO: Change to cartesian planning
                (plan_to_final, fraction) = self.move_group.compute_cartesian_path([p_base.pose], 0.01, 0)

                self.move_group.clear_pose_targets()

                if plan_to_final.joint_trajectory.points:
                # if plan_to_final.joint_trajectory.points and fraction != 1:

                    # Check id plan to offset grasp pose is valid
                    self.move_group.set_start_state(self.move_home_robot_state)
                    # self.move_group.set_pose_target(p_base_offset)
                    # plan_offset = self.move_group.plan()

                    # TODO: Change to cartesian planning
                    # (plan_offset, fraction) = self.move_group.compute_cartesian_path([p_base_offset.pose], 0.01, 0)
                    (plan_offset, fraction) = self.move_group.compute_cartesian_path([p_base_offset.pose, p_base.pose], 0.01, 0)
                    if fraction == 1:
                        self.move_group.clear_pose_targets()

                        if plan_offset.joint_trajectory.points:
                        # if plan_offset.joint_trajectory.points and fraction != 1:
                            # If so, we've found the grasp to use
                            final_grasp_pose = p_base
                            final_grasp_pose_offset = p_base_offset
                            robot_pose = p_base_robot
                            rospy.loginfo("Final grasp found!")
                            # rospy.loginfo(" Angle: %.4f",  theta_approach)
                            # Only display the grasp being used
                            poses = [poses[-1]]
                            break
                        else:
                            rospy.loginfo("Invalid path to final")
                            num_bad_plan += 1
                    else:
                        rospy.loginfo("Invalid path to final")
                        num_bad_plan += 1
                else:
                    rospy.loginfo("Invalid path to offset")
                    num_bad_plan += 1
            else:
                rospy.loginfo("Invalid angle of: " + str(theta_approach) + " deg")
                num_bad_angle += 1

        # Publish grasp pose arrows
        posearray = PoseArray()
        posearray.poses = poses
        posearray.header.frame_id = "base_link"
        self.pose_publisher.publish(posearray)

        #print("final_grasp_pose", final_grasp_pose)
        # rospy.loginfo("# bad angle: " + str(num_bad_angle))
        # rospy.loginfo("# bad plan: " + str(num_bad_plan))

        if not final_grasp_pose:
            plan_offset = 0
            rospy.loginfo("No valid grasp found!")

        return final_grasp_pose_offset, plan_offset, final_grasp_pose, robot_pose

    # Use class variables to move to a pose
    def move_to_position(self, grasp_pose, plan=None):
        attempted = move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, grasp_pose, plan, no_confirm=self.dont_display_plan)

        return attempted

    # Use class variables to move to a pose
    def move_to_cartesian_position(self, grasp_pose, plan=None):
        successful = False
        valid_path = False

        if plan == None:
            (plan, fraction) = self.move_group.compute_cartesian_path([grasp_pose.pose], 0.1, 0)
            if fraction != 1:
                rospy.logwarn("Invalid Path")
                plan = None
            else:
                # Show plan to check with user
                valid_path = self.user_check_path(grasp_pose, plan)
        else:
            valid_path = True

        # If user confirms the path
        if valid_path:
            self.move_group.execute(plan, wait=True)
            successful = True
        else:
            rospy.loginfo("Invalid Path")
            successful = False

        return successful

    # Visualise path
    def user_check_path(self, grasp_pose, plan):

        run_flag = "d"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y or n]? or display path again [d to display]:")

        if run_flag == "y":
            self.move_group.clear_pose_targets()
            return True
        elif run_flag == "n":
            self.move_group.clear_pose_targets()
            return False

    # Use class variables to move to a joint angle pose
    def move_to_joint_position(self, joint_array, plan=None):
        move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, joint_array, plan, no_confirm=self.dont_display_plan)

    # Publish a msg to the gripper
    def command_gripper(self, grip_msg):
        self.gripper_pub.publish(grip_msg)
        
    def gripper_state_callback(self, data):
        self.gripper_data = data

    # Defines post grasp lift pose
    def lift_up_pose(self):
        lift_dist = 0.1
        new_pose = self.move_group.get_current_pose()
        new_pose.pose.position.z += lift_dist
        return new_pose

    def run_motion(self, state, final_grasp_pose_offset, plan_offset, final_grasp_pose):
        # Set based on state to either box
        # drop_joints = self.drop_joints_no_box[state]if self.no_boxes else self.drop_joints[state]
        # LOW
        # drop_joints = [0.0, -1.6211016813861292, -1.9219277540790003, -1.166889492665426, 1.5740699768066406, -4.7985707418263246e-05]
        # HIGH
        drop_joints = [0.0006587578682228923, -1.6023038069354456, -1.6482232252704065, -1.4663317839251917, 1.5763914585113525, 0.0005153216770850122]

        # Joint sequence
        joints_to_move_to = [self.move_home_joints, self.stable_test_joints, self.move_home_joints]

        # Move home
        rospy.loginfo("Moving home...")
        self.move_group.set_start_state_to_current_state()
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.2)

        # Grab object
        rospy.loginfo("Moving to position...")
        # offset_attempted = self.move_to_position(final_grasp_pose_offset, plan_offset)
        offset_attempted = self.move_to_position(final_grasp_pose, plan_offset)
        rospy.sleep(0.2)

        # final_attempted = self.move_to_position(final_grasp_pose)
        # rospy.sleep(0.2)
        self.command_gripper(close_gripper_msg())
        rospy.sleep(1)

        # Lift up
        self.move_to_position(self.lift_up_pose())
        rospy.sleep(0.2)

        attempted_grasp = True

        # if offset_attempted and final_attempted:
        if offset_attempted:
            # Check grasp success
            # Success
            if self.check_grasp_success():
                rospy.loginfo("Robot has grasped the object!")
                success = 1
                # Stability check 
                rospy.loginfo("Checking stability...")
                for joints in joints_to_move_to:
                    rospy.sleep(0.1)
                    self.move_to_joint_position(joints)
                rospy.sleep(3)

                # Check if still in gripper (stable grasp)
                if self.check_grasp_success():
                    stable_success = 1
                    # Drop object
                    self.move_to_joint_position(drop_joints)
                else:
                    stable_success = 0
                    rospy.loginfo("Grasp was not stable!")
            # Fail
            else:
                rospy.loginfo("Robot has missed/dropped object!")
                success = 0
                stable_success = 0
        else:
            attempted_grasp = False
            rospy.loginfo("Robot could not plan!")
            success = 0
            stable_success = 0

        # Open gripper
        self.command_gripper(open_gripper_msg())
        rospy.sleep(0.5)

        # Move home    
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.2)

        return success, stable_success, attempted_grasp

    def run_cart_motion(self, state, final_grasp_pose_offset, final_grasp_pose, offset_plan):
        # Set based on state to either box
        # drop_joints = self.drop_joints_no_box[state]if self.no_boxes else self.drop_joints[state]
        drop_joints = [0.0, -1.6211016813861292, -1.9219277540790003, -1.166889492665426, 1.5740699768066406, -4.7985707418263246e-05]

        # Joint sequence
        joints_to_move_to = [self.move_home_joints, self.stable_test_joints, self.move_home_joints]

        successful_to_offset = False
        plan_attempt = 0

        while not successful_to_offset:
            # Grab object
            rospy.loginfo("Move to offset")
            rospy.sleep(2)
            successful_to_offset = self.move_to_cartesian_position(final_grasp_pose_offset) # Transition to cart
            rospy.sleep(2)
            plan_attempt = plan_attempt + 1
            if successful_to_offset:
                rospy.sleep(2)
            else:
                # Give up after 3 plans
                if plan_attempt > 3:
                    break

        # If success found
        if successful_to_offset:
            rospy.loginfo("Move to final position")
            self.move_to_cartesian_position(final_grasp_pose)
            rospy.sleep(2)
            self.command_gripper(close_gripper_msg())
            rospy.sleep(1)
            attempted_grasp = True

            # Lift up
            rospy.loginfo("Lift up")
            self.move_to_position(self.lift_up_pose())
            rospy.sleep(0.2)

            # Check grasp success
            # Success
            if self.check_grasp_success():
                rospy.loginfo("Robot has grasped the object!")
                success = 1
                # Stability check 
                rospy.loginfo("Checking stability...")
                for joints in joints_to_move_to:
                    rospy.sleep(0.1)
                    self.move_to_joint_position(joints)
                rospy.sleep(3)

                # Check if still in gripper (stable grasp)
                if self.check_grasp_success():
                    stable_success = 1
                else:
                    stable_success = 0

                if stable_success:
                    # Drop object
                    self.move_to_joint_position(drop_joints)

            # Fail
            else:
                rospy.loginfo("Robot has missed/dropped object!")
                success = 0
                stable_success = 0
        else:
            attempted_grasp = False
            rospy.loginfo("Robot could not plan!")
            success = 0
            stable_success = 0

        # Open gripper
        self.command_gripper(open_gripper_msg())
        rospy.sleep(0.5)

        # Move home    
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.2)

        return success, stable_success, attempted_grasp

    def check_grasp_success(self):
        if self.gripper_data.gOBJ == 2 and self.gripper_data.gPO < 255:
            # Object in gripper
            return True
        else:
            return False
        # if self.gripper_data.gOBJ == 3:
        #     # Object NOT in gripper        

    # Takes a joint array and returns a robot state
    def get_robot_state(self, joint_list):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint',  'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        joint_state.position = joint_list
        robot_state = RobotState()
        robot_state.joint_state = joint_state
        return robot_state

    def data_saver(self, object_id, rgb_image, depth_image, point_cloud, trans, rot, cam_info, ee_pose, robot_pose, success, stable_success):
        time_now = rospy.Time.from_sec(time.time())
        header = Header()
        header.stamp = time_now

        # TODO: ints
        self.bag.write('time', header)
        self.bag.write('object_id', intToMsg(object_id))
        self.bag.write('rgb_image', rgb_image)  # Save an image
        self.bag.write('depth_image', depth_image)
        self.bag.write('point_cloud', point_cloud)
        self.bag.write('trans', floatArrayToMsg(trans))
        self.bag.write('rot', floatArrayToMsg(rot))
        self.bag.write('cam_info', cam_info)
        self.bag.write('ee_pose', ee_pose)
        self.bag.write('robot_pose', robot_pose) 
        self.bag.write('success', intToMsg(success))
        self.bag.write('stable_success', intToMsg(stable_success))

        rospy.loginfo("Object ID: %d", object_id)
        rospy.loginfo("Success: %d", success)
        rospy.loginfo("Stable Success: %d", stable_success)

    def get_transform(self):
        got_trans = False
        while not got_trans:
            try:
                trans = self.tf_buffer.lookup_transform('base_link', 'camera_link', rospy.Time.now(), rospy.Duration(1.0))
                got_trans = True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep(1)
        return trans

    def main(self):
        # Startup
        rate = rospy.Rate(1)
        while not self.gripper_data and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect...")
            rospy.sleep(1)

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
        rospy.sleep(1)
        rospy.loginfo("Moved to Home Position")
 
        #: Generate an intial box to grab from based on # of objects in the box
        self.state = State.RIGHT_TO_LEFT

        #: Set initial workspace based on current state
        # ws_curr = self.workspaces[self.state]
        ws_curr = self.workspace

        #: Init counter of failed grasps
        # failed_grasps = 0 <maybe?>

        # Counters
        view_counter = 0
        # attempt_counter = 0.0
        # success_counter = 0.0

        rospy.set_param("/detect_grasps/workspace", ws_curr)

        # TEST
        # Create poses for grasp and pulled back (offset) grasp
        p_test = PoseStamped()

        p_test.header.frame_id = "base_link"

        p_test.pose.position.x = -0.499575960768
        p_test.pose.position.y = 0.0349784804409
        p_test.pose.position.z = 0.039447361082

        p_test.pose.orientation.x = -0.166033880986
        p_test.pose.orientation.y = 0.496979279168
        p_test.pose.orientation.z = 0.264681925683
        p_test.pose.orientation.w = 0.809560266231

        rospy.loginfo("Moving to test pose...")
        self.move_to_position(p_test)
        rospy.sleep(5)

        p_test.pose.position.x = -0.561048865731
        p_test.pose.position.y = -0.00929307166114
        p_test.pose.position.z = 0.189397724969

        rospy.loginfo("Moving to test pose...")
        self.move_to_position(p_test)
        rospy.sleep(5)

        while not rospy.is_shutdown():
            
            # Generate a point cloud from several readings
            self.agile_state = AgileState.WAIT_FOR_ONE
            
            # Point cloud from two views?
            # point_cloud = self.generate_pcl(int(self.state), view_counter) #### : set generate_pcl input based on state
            # self.PCL_stitched_publisher.publish(point_cloud.cloud)

            # -- Capture point cloud --
            view_joint = self.view_joints[view_counter]
            # Increment view counter
            view_counter = view_counter + 1
            # view_counter = view_counter % 4
            view_counter = view_counter % 3

            # Move UR5
            move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, view_joint, no_confirm=True)
            rospy.sleep(1)
            rospy.loginfo("Generating point cloud...")

            valid_pointcloud = False
            while not valid_pointcloud:
                # Capture
                time_start = rospy.Time.now()
                rospy.sleep(1)
                raw_point_cloud = self.pcl_rosmsg
                # point_cloud = copy.deepcopy(raw_point_cloud)
                rospy.sleep(1)

                # Publish to assembler
                # self.PCL_publisher.publish(raw_point_cloud)
                # Assemble 
                # resp = self.assemble_scans(time_start, rospy.Time.now())

                # Transform
                trans = self.get_transform()
                point_cloud = do_transform_cloud(raw_point_cloud, trans)

                # Point Cloud
                rospy.loginfo("Point cloud generated")

                # if len(resp.cloud.data) > 0:
                if len(point_cloud.data) > 0:
                    # Publish to agilegrasp
                    # self.PCL_stitched_publisher.publish(resp.cloud)
                    self.PCL_stitched_publisher.publish(point_cloud)
                    valid_pointcloud = True
                    rospy.loginfo("VALID POINTCLOUD")
                else:
                    rospy.loginfo("EMPTY POINTCLOUD")
            
            # Capture RGB and Depth
            rgb_image = self.rgb_image
            depth_image = self.depth_image

            # Convert to O3D
            # o3d_point_cloud = self.convertCloudFromRosToOpen3d(point_cloud)
            # o3d.visualization.draw_geometries([o3d_point_cloud])

            rospy.sleep(1)

            self.test_publisher.publish(point_cloud)

            # Listen to tf
            self.tf_listener_.waitForTransform("/base_link", "/camera_link", rospy.Time(), rospy.Duration(4))
            (trans, rot) = self.tf_listener_.lookupTransform('/base_link', '/camera_link', rospy.Time(0))

            # Camera intrinsics
            cam_info = self.cam_info

            # Wait for a valid reading from agile grasp
            while not rospy.is_shutdown() and self.agile_state is not AgileState.READY:
                rospy.loginfo("Waiting for agile grasp...")
                self.command_gripper(close_gripper_msg())
                rospy.sleep(0.2)
                self.command_gripper(open_gripper_msg())
                rospy.sleep(3)
            rospy.loginfo("Grasp pose detection complete")

            # Find best grasp from reading
            rospy.loginfo("Finding valid grasp...")
            final_grasp_pose_offset, offset_plan, final_grasp_pose, robot_pose = self.find_best_grasp(self.agile_data, self.choose_random)
            # final_grasp_pose_offset, final_grasp_pose, offset_plan = self.find_best_cart_grasp(self.agile_data, self.choose_random)
            
            if final_grasp_pose:
                rospy.loginfo("Grasp found! Executing grasp")
                #Run the current motion on it 
                success, stable_success, attempted_grasp = self.run_motion(self.state, final_grasp_pose_offset, offset_plan, final_grasp_pose)
                # success, stable_success, attempted_grasp = self.run_cart_motion(self.state, final_grasp_pose_offset, final_grasp_pose, offset_plan)

                if attempted_grasp:
                    # Counter
                    self.attempt_counter = self.attempt_counter + 1
                    if stable_success:
                        self.success_counter = self.success_counter + 1

                    # Determine ur5 pose -> final_grasp_pose_offset
                    
                    # Save data
                    self.data_saver(self.object_id, rgb_image, depth_image, point_cloud, trans, rot, cam_info, final_grasp_pose, robot_pose, success, stable_success)

                    # Close bag
                    self.bag.close()
                    
                    # Adjust ratio
                    self.success_ratio = self.success_counter / self.attempt_counter

                    if (self.attempt_counter % 50 == 0):
                        # Change object
                        rospy.loginfo("\nPrevious Object ID: %d", self.object_id)
                        self.object_id = self.object_id + 1
                        rospy.loginfo("\nNew Object ID: %d", self.object_id)
                        raw_input("\n\n\nPlace object and press enter...")
                        break
                    elif (self.attempt_counter % 25 == 0):
                        # Change Change to random
                        raw_input("\n\n\nLeave object in random position and press enter...")

                    # Open new bag
                    self.bag = rosbag.Bag('/home/acrv/new_ws/src/grasp_executor/scripts/grasping_demo/grasp_data_bags/data_' + str(int(math.floor(time.time()))) + ".bag", 'w')
                else:
                    rospy.loginfo("Did not attempt!")
            else:
                rospy.loginfo("No pose target generated!")

            # Move back home
            rospy.loginfo("Moving home...")
            self.move_group.set_start_state_to_current_state()
            self.move_to_joint_position(self.move_home_joints)
            rospy.sleep(1)
            self.command_gripper(open_gripper_msg())
            rospy.loginfo("Adjust object")
            rospy.sleep(3)

            # Print progress
            rospy.loginfo(" -------------------------- ")
            rospy.loginfo("Attempt counter: %f", self.attempt_counter)
            rospy.loginfo("Success counter: %f", self.success_counter)
            rospy.loginfo(" -------------------------- ")
            
            rate.sleep()

            #### Future work:
            #
            # If a valid grasp pose wasn't found, increment fail counter
            # If it was found, perform the motion
            #     If the motion failed (was dropped) increment counter
            #
            # 1. Check the counter. If the threshold has been hit, try to switch boxes (and reset counter)
            # 2. Check that there are still objects to grab. If there arent, switch boxes
            #
            #
            # Maybe there sould be a function to swap boxes (that can be used to initialize as well?)
            # As the state and ws need to both be swapped
            #
            # Also add an open at the end of the iteration, to make sure the gripper always ends open
            # Then sleep

    def convertCloudFromRosToOpen3d(self, ros_cloud):
        # Get cloud data from ros_cloud
        field_names=[field.name for field in ros_cloud.fields]
        cloud_data = list(pc2.read_points(ros_cloud, skip_nans=True, field_names = field_names))

        # Check empty
        open3d_cloud = o3d.geometry.PointCloud()
        if len(cloud_data)==0:
            print("Converting an empty cloud")
            return None

        # Set open3d_cloud
        if "rgba" in field_names:
            IDX_RGB_IN_FIELD=3 # x, y, z, rgb
            
            # Get xyz
            xyz = [(x,y,z) for x,y,z,rgb in cloud_data ] # (why cannot put this line below rgb?)

            # Get rgb
            # Check whether int or float
            if type(cloud_data[0][IDX_RGB_IN_FIELD])==float: # if float (from pcl::toROSMsg)
                rgb = [convert_rgbFloat_to_tuple(rgb) for x,y,z,rgb in cloud_data ]
            else:
                rgb = [convert_rgbUint32_to_tuple(rgb) for x,y,z,rgb in cloud_data ]

            # combine
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))
            open3d_cloud.colors = o3d.utility.Vector3dVector(np.array(rgb)/255.0)
        else:
            xyz = [(x,y,z) for x,y,z in cloud_data ] # get xyz
            open3d_cloud.points = o3d.utility.Vector3dVector(np.array(xyz))

        # return
        return open3d_cloud


if __name__ == '__main__':
    try:
        grasper = GraspExecutor()
        grasper.main()
    except KeyboardInterrupt:
        pass



# [-1.6482232252704065, -1.6023038069354456, 0.0006587578682228923, -1.4663317839251917, 1.5763914585113525, 0.0005153216770850122]
[0.0006587578682228923, -1.6023038069354456, -1.6482232252704065, -1.4663317839251917, 1.5763914585113525, 0.0005153216770850122]

# ['shoulder_pan_joint', 'shoulder_lift_joint',  'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

# [elbow_joint, shoulder_lift_joint, shoulder_pan_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint]
# [-1.82951528230776, -0.631601635609762, -1.1746083394825746e-05, -2.1099846998797815, 1.569976568222046, -4.7985707418263246e-05]
# [-1.1746083394825746e-05, -0.631601635609762, -1.82951528230776, -2.1099846998797815, 1.569976568222046, -4.7985707418263246e-05]

# [-0.23039132753481084, -2.110682789479391, 0.0, -2.8553083578692835, 1.5699286460876465, -3.5587941304981996e-05]
# [0.0, -2.110682789479391, -0.23039132753481084, -2.8553083578692835, 1.5699286460876465, -3.5587941304981996e-05]

# [-0.40200978914369756, -2.0729802290545862, -0.5584028402911585, -2.636850659047262, 2.227617025375366, 0.8789638876914978]
# [-0.5584028402911585, -2.0729802290545862, -0.40200978914369756, -2.636850659047262, 2.227617025375366, 0.8789638876914978]

# [-0.403877083455221, -2.1710355917560022, 0.7862164974212646, -2.9715493361102503, 1.0643872022628784, -0.5000680128680628]
# [0.7862164974212646, -2.1710355917560022, -0.403877083455221, -2.9715493361102503, 1.0643872022628784, -0.5000680128680628]

# [-1.9219277540790003, -1.6211016813861292, 0.0, -1.166889492665426, 1.5740699768066406, -4.7985707418263246e-05]
# [0.0, -1.6211016813861292, -1.9219277540790003, -1.166889492665426, 1.5740699768066406, -4.7985707418263246e-05]