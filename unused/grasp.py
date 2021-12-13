#!/usr/bin/env python
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

import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatusArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg
from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from util import dist_to_guess, vector3ToNumpy


from pyquaternion import Quaternion

import pdb
from enum import Enum

class State(Enum):
    FIRST_GRAB=1
    SECOND_GRAB=2
    FINISHED=3

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
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        self.view_home_joints = [0.24985386431217194, -0.702608887349264, -2.0076406637774866, -1.7586587111102503, 1.5221580266952515, 0.25777095556259155]
        self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
        self.drop_object_joints = [0.14647944271564484, -1.8239172140704554, -1.0428651014911097, -1.8701766172992151, 1.6055123805999756, 0.03247687593102455]
        self.deliver_object_joints = [-0.5880172888385218, -2.375404659901754, -0.8875716368304651, -1.437070671712057, 1.6041597127914429, 0.032297488301992416]

        self.move_home_robot_state = self.get_robot_state(self.move_home_joints)

        self.agile_data = 0
        self.agile_state = AgileState.WAIT_FOR_ONE

        rospy.Subscriber("/detect_grasps/grasps", GraspListMsg, self.agile_callback)

    def agile_callback(self, data):
        self.agile_data = data

        self.agile_state = AGILE_STATE_TRANSITION[self.agile_state]


    def find_best_grasp(self, data):
        max_angle = 90
        final_grasp_pose = 0
        final_grasp_pose_offset = 0

        num_bad_angle = 0
        num_bad_plan = 0

        poses = []
        # Sort grasps by quality
        data.grasps.sort(key=lambda x : x.score, reverse=True)
        rospy.loginfo("Grasps Sorted!")
        
        for g in data.grasps:
            R = np.zeros((3,3))
            R[:, 0] = vector3ToNumpy(g.approach)
            R[:, 1] = vector3ToNumpy(g.axis)
            R[:, 2] = np.cross(vector3ToNumpy(g.approach), vector3ToNumpy(g.axis))

            q = Quaternion(matrix=R)
            position =  g.surface
            rospy.loginfo("Grasp cam orientation found!")

            p_cam = PoseStamped()

            offset_dist = 0.1

            p_cam.pose.position.x = position.x 
            p_cam.pose.position.y = position.y 
            p_cam.pose.position.z = position.z 

            p_cam.pose.orientation.x = q[1]
            p_cam.pose.orientation.y = q[2]
            p_cam.pose.orientation.z = q[3]
            p_cam.pose.orientation.w = q[0]

            p_cam_offset = copy.deepcopy(p_cam)
            p_cam_offset.pose.position.x -= g.approach.x *offset_dist
            p_cam_offset.pose.position.y -= g.approach.y *offset_dist
            p_cam_offset.pose.position.z -= g.approach.z *offset_dist

            self.tf_listener_.waitForTransform("/camera_link", "/base_link", rospy.Time(), rospy.Duration(4))

            p_cam.header.frame_id = "camera_link"
            p_base = self.tf_listener_.transformPose("/base_link", p_cam)

            p_cam_offset.header.frame_id = "camera_link"
            p_base_offset = self.tf_listener_.transformPose("/base_link", p_cam_offset)

            poses.append(copy.deepcopy(p_base.pose))

            transform_matrix = self.tf_listener_.asMatrix("/base_link", p_cam.header)
            approach_base = np.matmul(transform_matrix, np.array([g.approach.x, g.approach.y, g.approach.z, 1]).T)
            approach_base = approach_base[:3]
            approach_base = approach_base / np.linalg.norm(approach_base)

            theta_approach = np.arccos(np.dot(approach_base, np.array([0,0,-1])))*180/np.pi

            rospy.loginfo("Grasp base orientation found")   

            if theta_approach < max_angle:
                
                self.move_group.set_start_state(self.move_home_robot_state)
                self.move_group.set_pose_target(p_base)
                plan_to_final = self.move_group.plan()

                self.move_group.clear_pose_targets()
                if plan_to_final.joint_trajectory.points:

                    self.move_group.set_start_state(self.move_home_robot_state)
                    self.move_group.set_pose_target(p_base_offset)
                    plan_offset = self.move_group.plan()

                    if plan_offset.joint_trajectory.points:
                        final_grasp_pose = p_base
                        final_grasp_pose_offset = p_base_offset
                        rospy.loginfo("Final grasp found!")
                        rospy.loginfo(" Angle: %.4f",  theta_approach)
                        poses = [poses[-1]]
                        break
                    else:
                        rospy.loginfo("Invalid path")
                        num_bad_plan += 1

                else:
                    rospy.loginfo("Invalid path")
                    num_bad_plan += 1
            else:
                rospy.loginfo("Invalid angle of: " + str(theta_approach) + " deg")
                num_bad_angle += 1

        posearray = PoseArray()
        posearray.poses = poses
        posearray.header.frame_id = "base_link"

        print("final_grasp_pose", final_grasp_pose)

        self.pose_publisher.publish(posearray)

        rospy.loginfo("# bad angle: " + str(num_bad_angle))
        rospy.loginfo("# bad plan: " + str(num_bad_plan))

        if not final_grasp_pose:
            plan_offset = 0

        return final_grasp_pose_offset, plan_offset, final_grasp_pose

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

    def command_gripper(self, grip_msg):
        self.gripper_pub.publish(grip_msg)
        
    def gripper_state_callback(self, data):
        self.gripper_data = data

    def lift_up_pose(self):
        lift_dist = 0.05

        new_pose = self.move_group.get_current_pose()

        new_pose.pose.position.z += lift_dist

        return new_pose

    def get_drop_pose(self):
        drop = PoseStamped()

        drop.pose.position.x = -0.450
        drop.pose.position.y = -0.400
        drop.pose.position.z = 0.487

        return drop

    def run_motion(self, state, final_grasp_pose_offset, plan_offset, final_grasp_pose):
        if state == State.FIRST_GRAB:
            self.move_group.set_start_state_to_current_state()
            self.move_to_joint_position(self.move_home_joints)
            self.move_to_position(final_grasp_pose_offset, plan_offset)
            self.move_to_position(final_grasp_pose)
            self.command_gripper(close_gripper_msg())
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
        elif state == State.SECOND_GRAB:
            self.move_group.set_start_state_to_current_state()
            self.move_to_joint_position(self.move_home_joints)
            self.move_to_position(final_grasp_pose_offset, plan_offset)
            self.move_to_position(final_grasp_pose)
            self.command_gripper(close_gripper_msg())
            self.move_to_position(self.lift_up_pose())

            rospy.sleep(1)
            if self.gripper_data.gOBJ == 3:
                rospy.loginfo("Robot has missed/dropped object!")
                self.move_to_joint_position(self.move_home_joints)
                self.move_to_joint_position(self.view_home_joints)
            else:
                # Go to move home position using joint
                self.move_to_joint_position(self.move_home_joints)

                self.move_to_joint_position(self.deliver_object_joints)
                self.command_gripper(open_gripper_msg())
                self.move_to_joint_position(self.move_home_joints)

                self.state = State.FINISHED
            rospy.sleep(2)
        else:
            rospy.loginfo("Robot has finished!")    

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
        rate = rospy.Rate(1)
        # Startup

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
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.1)
        rospy.loginfo("Moved to Home Position")
        self.move_to_joint_position(self.view_home_joints)
        rospy.sleep(0.1)
        rospy.loginfo("Moved to View Position")

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
                self.run_motion(self.state, final_grasp_pose_offset, plan_offset, final_grasp_pose)
            else:
                rospy.loginfo("No pose target generated!")

            if self.state == State.FINISHED:
                rospy.loginfo("Task complete!")
                rospy.spin()
            
            rate.sleep()


if __name__ == '__main__':
    try:
        grasper = GraspExecutor()
        grasper.main()
    except KeyboardInterrupt:
        pass
