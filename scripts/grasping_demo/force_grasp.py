#!/usr/bin/env python
# Imports
import rospy
import sys
from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray
from std_msgs.msg import Header
import numpy as np
import tf
from tf import TransformListener
import copy 
from time import sleep
import roslaunch
import math

import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
from scripts.gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from scripts.util import dist_to_guess, vector3ToNumpy

from pyquaternion import Quaternion

import pdb
from enum import Enum

# Enums
class State(Enum):
    FIRST_GRAB=1
    SECOND_GRAB=2
    FINISHED=3

class AgileState(Enum):
    RESET = 0
    WAIT_FOR_ONE = 1
    READY = 2

# Transitions
AGILE_STATE_TRANSITION = {
    AgileState.RESET: AgileState.WAIT_FOR_ONE,
    AgileState.WAIT_FOR_ONE: AgileState.READY,
    AgileState.READY: AgileState.READY
}

GRAB_THRESHOLD = 8 # Newtons
RELEASE_THRESHOLD = 8 # Newtons

# Grasp Class
class GraspExecutor:
    # Initialisation
    def __init__(self):
        # Initialisation
        rospy.init_node('grasp_executor', anonymous=True)

        self.tf_listener_ = TransformListener()
        self.launcher = roslaunch.scriptapi.ROSLaunch()
        self.launcher.start()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.pose_publisher = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)

        self.box_drop = self.get_drop_pose()

        self.state = State.FIRST_GRAB

        self.gripper_data = 0
        self.force_sub = rospy.Subscriber('robotiq_ft_wrench', WrenchStamped, self.force_callback)
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.latest_force = 0.0

        # Hard-coded joint values
        self.view_home_joints = [0.24985386431217194, -0.702608887349264, -2.0076406637774866, -1.7586587111102503, 1.5221580266952515, 0.25777095556259155]
        self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
        self.drop_object_joints = [0.14647944271564484, -1.8239172140704554, -1.0428651014911097, -1.8701766172992151, 1.6055123805999756, 0.03247687593102455]
        self.deliver_object_joints = [-0.5880172888385218, -2.375404659901754, -0.8875716368304651, -1.437070671712057, 1.6041597127914429, 0.032297488301992416]

        self.move_home_robot_state = self.get_robot_state(self.move_home_joints)

        # Hard-code corner positions
        # Start Top Right (robot perspective) and go around clockwise
        corner_1 = [-825,235]
        corner_2 = [-410,235]
        corner_3 = [-410,-100]
        corner_4 = [-825,-100]
        self.corner_pos_list = [corner_1, corner_2, corner_3, corner_4]

        # AgileGrasp data
        self.agile_data = 0
        self.agile_state = AgileState.WAIT_FOR_ONE
        rospy.Subscriber("/detect_grasps/grasps", GraspListMsg, self.agile_callback)

    def force_callback(self, wrench_msg):
        self.latest_force = abs(wrench_msg.wrench.force.z)
    
    def agile_callback(self, data):
        # Callback function for agilegrasp data
        self.agile_data = data
        self.agile_state = AGILE_STATE_TRANSITION[self.agile_state]

    def gripper_state_callback(self, data):
        # function called when gripper data is received
        self.gripper_data = data

    def find_best_grasp(self, data):
        # Determine the best grasp from agilegrasp grasp list
        # Angle at which grasps are performed
        grasp_angle = 30
        # Initialise values
        final_grasp_pose = 0
        final_grasp_pose_offset = 0
        num_bad_angle = 0
        num_bad_plan = 0
        # Grasp pose list
        poses = []
        # Sort grasps by quality
        data.grasps.sort(key=lambda x : x.score, reverse=True)
        rospy.loginfo("Grasps Sorted!")
        
        for g in data.grasps:
            # Position of grasp (On the object surface)
            position =  g.surface
            rospy.loginfo("Grasp cam orientation found!")
            # Create pose in camera frame
            p_cam = PoseStamped()
            # Grasp pose offset distance
            offset_dist = 0.1
            # Add position of agilegrasp grasp to pose
            p_cam.pose.position.x = position.x 
            p_cam.pose.position.y = position.y 
            p_cam.pose.position.z = position.z
            # Listen to tf
            self.tf_listener_.waitForTransform("/camera_link", "/base_link", rospy.Time(), rospy.Duration(4))
            # Transform pose from camera frame to base frame
            p_cam.header.frame_id = "camera_link"
            p_base = self.tf_listener_.transformPose("/base_link", p_cam)

            # Find nearest corner
            nearest_corner = self.find_nearest_corner(p_base)
            # Convert positions to 2D
            corner_pos = self.corner_pos_list(nearest_corner) # [x,y]
            grasp_pos = [p_base.pose.position.x, p_base.pose.position.y] # [x,y]

            # Find approach angle
            z_angle, offset_pos = self.calculate_approach(grasp_pos, corner_pos, -offset_dist)

            # Print results
            rospy.loginfo("Nearest corner: %d", nearest_corner)
            rospy.loginfo("Corner Position - X: %d Y:%d", corner_pos[0], corner_pos[1])
            rospy.loginfo("Grasp Position - X: %d Y:%d", grasp_pos[0], grasp_pos[1])
            rospy.loginfo("Z-angle: %d", z_angle)

            # Find angle of gripper to the ground from hard-coded value
            y_angle = np.deg2rad(grasp_angle)

            # Generate pose
            p_base = self.generate_push_pose(p_base, offset_pos[0], offset_pos[0], y_angle, z_angle)

            # Create offset pose
            p_base_offset = copy.deepcopy(p_base)
            p_base_offset.pose.position.x -= offset_pos[0] *offset_dist
            p_base_offset.pose.position.y -= offset_pos[1] *offset_dist

            # Add pose to pose list
            poses.append(copy.deepcopy(p_base.pose))
            
            # Check path planning from home state to final grasp pose
            self.move_group.set_start_state(self.move_home_robot_state)
            self.move_group.set_pose_target(p_base)
            plan_to_final = self.move_group.plan()
            # Clear target
            self.move_group.clear_pose_targets()
            # If plan is valid
            if plan_to_final.joint_trajectory.points:
                # Check path planning from home state to offset grasp pos
                self.move_group.set_start_state(self.move_home_robot_state)
                self.move_group.set_pose_target(p_base_offset)
                plan_offset = self.move_group.plan()
                # If we can move to offset position
                if plan_offset.joint_trajectory.points:
                    # Use current poses as final grasp poses
                    final_grasp_pose = p_base
                    final_grasp_pose_offset = p_base_offset
                    rospy.loginfo("Final grasp found!")
                    poses = [poses[-1]]
                    break
                else:
                    rospy.loginfo("Invalid path")
                    num_bad_plan += 1

            else:
                rospy.loginfo("Invalid path")
                num_bad_plan += 1

        # Create pose array and add valid grasps
        posearray = PoseArray()
        posearray.poses = poses
        posearray.header.frame_id = "base_link"

        rospy.loginfo("final_grasp_pose", final_grasp_pose)
        # Publish grasp array
        self.pose_publisher.publish(posearray)
        # Print number of invalid plans
        rospy.loginfo("# bad angle: " + str(num_bad_angle))
        rospy.loginfo("# bad plan: " + str(num_bad_plan))

        # If there is no valid grasp
        if not final_grasp_pose:
            # Offset value is 0
            plan_offset = 0

        return final_grasp_pose_offset, plan_offset, final_grasp_pose

    def find_nearest_corner(self, p_base):
        # Find the nearest corner to the 
        grasp_x = p_base.pose.position.x
        grasp_y = p_base.pose.position.y
        grasp_pos = [grasp_x, grasp_y]

        distance_list = [0,0,0,0]
        # Calculate distance between the grasp point and the corners
        for i in range(len(self.corner_pos_list)):
            corner_pos = self.corner_pos_list[i]
            distance_list[i] = math.dist(grasp_pos, corner_pos)
        # Nearest corner 
        nearest_corner = distance_list.index(min(distance_list))

        return nearest_corner

    def calculate_approach(self, start_pos, final_pos, distance):
        # start_pos [x,y]
        # final_pos [x,y]
        x_diff = final_pos[0] - start_pos[0]
        y_diff = final_pos[1] - start_pos[1]
        # Angle of the gripper to the corner (in z-axis)
        z_angle = atan2(y_diff, x_diff)

        # Calculate offset position
        v = np.array([x_diff, y_diff])
        v_magnitude = math.sqrt(x_diff*x_diff + y_diff*y_diff)
        u = v / v_magnitude
        new_pos = np.array(start_pos) + distance*u

        rospy.loginfo("Start Position: %d", start_pos)
        rospy.loginfo("Final Position: %d", final_pos)
        rospy.loginfo("New Position: %d", new_pos)

        return z_angle, new_pos

    def generate_push_pose(self, current_pose, x_pos, y_pos, y_angle, z_angle):
        # Update pose in base frame
        # Position
        current_pose.pose.position.x += x_pos
        current_pose.pose.position.y += y_pos
        # Orientation
        # Create quaternion
        quaternion = tf.transformations.quaternion_from_euler(0, y_angle, z_angle)
        # Update values
        current_pose.pose.orientation.x = quaternion[1]
        current_pose.pose.orientation.y = quaternion[2]
        current_pose.pose.orientation.z = quaternion[3]
        current_pose.pose.orientation.w = quaternion[0]

        return current_pose

    def force_grasp(self, corner_pos):
        threshold = 1
        push_dist = 0.01
        # Check force feedback
        force_feedback = self.latest_force
        # While force less then threshold value
        while force_feedback < threshold:
            # Move towards corner
            new_pose = self.move_group.get_current_pose()
            # Find current 2D co-ordinates
            current_pos = [new_pose.pose.position.x, new_pose.pose.position.y]
            # Find approach angle
            z_angle, new_pos = self.calculate_approach(current_pos, corner_pos, push_dist)
            # Find angle of gripper to the ground from hard-coded value
            y_angle = np.deg2rad(grasp_angle)

            # Generate new pose
            new_pose = self.generate_push_pose(new_pos[0], new_pos[1], y_angle, z_angle)

            self.move_to_position(new_pose)
            # Check force feedback
            force_feedback = self.latest_force
        else:
            # Close gripper
            self.command_gripper(close_gripper_msg())

    def run_motion(self, state, final_grasp_pose_offset, plan_offset, final_grasp_pose):
        if state == State.FIRST_GRAB:
            self.move_group.set_start_state_to_current_state()
            self.move_to_joint_position(self.move_home_joints)
            self.move_to_position(final_grasp_pose_offset, plan_offset)
            self.move_to_position(final_grasp_pose)
            # force grasp
            self.force_grasp(self, final_grasp_pose)

            self.move_to_position(self.lift_up_pose())

            rospy.sleep(1)
            if self.gripper_data.gOBJ == 3:
                rospy.loginfo("Robot has missed/dropped object!")
                self.move_to_joint_position(self.move_home_joints)
                self.move_to_joint_position(self.view_home_joints)
            else:
                # Go to move home position using joint
                self.move_to_joint_position(self.move_home_joints)
                self.move_to_joint_position(self.drop_object_joints)
                self.command_gripper(open_gripper_msg())
                self.move_to_joint_position(self.move_home_joints)
                self.move_to_joint_position(self.view_home_joints)

                self.state = State.SECOND_GRAB

            rospy.sleep(2)
        else:
            rospy.loginfo("Robot has finished!") 

    def move_to_position(self, grasp_pose, plan=None):
        self.move_group.set_pose_target(grasp_pose)
        if not plan:
            plan = self.move_group.plan()

        run_flag = "d"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

        if run_flag =="y":
            self.move_group.execute(plan, wait=True)


        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def move_to_joint_position(self, joint_array, plan=None):
        self.move_group.set_joint_value_target(joint_array)
        if not plan:
            plan = self.move_group.plan()

        run_flag = "d"

        while run_flag == "d":
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            run_flag = raw_input("Valid Trajectory [y to run]? or display path again [d to display]:")

        if run_flag =="y":
            self.move_group.execute(plan, wait=True)


        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def get_drop_pose(self):
        drop = PoseStamped()

        drop.pose.position.x = -0.450
        drop.pose.position.y = -0.400
        drop.pose.position.z = 0.487

        return drop

    def command_gripper(self, grip_msg):
        # publish gripper message to gripper
        self.gripper_pub.publish(grip_msg)

    def lift_up_pose(self):
        # lift gripper up
        lift_dist = 0.05
        new_pose = self.move_group.get_current_pose()
        new_pose.pose.position.z += lift_dist
        return new_pose

    def get_robot_state(self, joint_list):
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint',  'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        joint_state.position = joint_list
        robot_state = RobotState()
        robot_state.joint_state = joint_state

        return robot_state

    def launch_pcl_process(self, pcl_node):
        pcl_process = self.launcher.launch(pcl_node)
        while not pcl_process.is_alive():
            rospy.sleep(0.1)
        return pcl_process

    def stop_pcl_process(self, pcl_process):
        pcl_process.stop()
        while pcl_process.is_alive():
            rospy.sleep(0.1)

    def main(self):
        # Set rate
        rate = rospy.Rate(1)
        # Gripper startup sequence
        while not self.gripper_data and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect")
            rospy.sleep(1)
        self.command_gripper(reset_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(activate_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(close_gripper_msg())
        rospy.sleep(.1)
        self.command_gripper(open_gripper_msg())
        rospy.sleep(.1)
        rospy.loginfo("Gripper active")

        # Go to move home position using joint
        # self.move_to_joint_position(self.move_home_joints)
        # rospy.sleep(0.1)
        # rospy.loginfo("Moved to Home Position")
        # self.move_to_joint_position(self.view_home_joints)
        # rospy.sleep(0.1)
        # rospy.loginfo("Moved to View Position")

        while not rospy.is_shutdown():
            # Boot up pcl
            pcl_node = roslaunch.core.Node('grasp_executor', 'pcl_preprocess_node.py')
            pcl_process = self.launch_pcl_process(pcl_node)

            #Wait for a valid reading from agile grasp
            self.agile_state = AgileState.RESET
            while self.agile_state is not AgileState.READY:
                rospy.loginfo("Waiting for agile grasp")
                rospy.sleep(2)
            
            rospy.loginfo("Grasp detection complete")
            #Stop pcl
            self.stop_pcl_process(pcl_process)

            #Find best grasp from reading
            final_grasp_pose_offset, plan_offset, final_grasp_pose = self.find_best_grasp(self.agile_data)

            if final_grasp_pose:
                # self.run_motion(self.state, final_grasp_pose_offset, plan_offset, final_grasp_pose)
                pass
            else:
                rospy.loginfo("No pose target generated!")

            if self.state == State.SECOND_GRAB:
                rospy.loginfo("Task complete!")
                rospy.spin()
            
            rate.sleep()


if __name__ == '__main__':
    try:
        grasper = GraspExecutor()
        grasper.main()
    except KeyboardInterrupt:
        pass