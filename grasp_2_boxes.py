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


import moveit_commander
import moveit_msgs.msg
from std_msgs.msg import Header
from moveit_msgs.msg import DisplayTrajectory, MoveGroupActionFeedback, RobotState
from sensor_msgs.msg import JointState, PointCloud2
from actionlib_msgs.msg import GoalStatusArray
from agile_grasp2.msg import GraspListMsg
from geometry_msgs.msg import PoseStamped, WrenchStamped, PoseArray
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg, _Robotiq2FGripper_robot_input as inputMsg

from gripper import open_gripper_msg, close_gripper_msg, activate_gripper_msg, reset_gripper_msg
from util import dist_to_guess, vector3ToNumpy, move_ur5
from grasp_executor.srv import PCLStitch



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
        rospy.loginfo("Node active!")

        #### Useful variables ####
        #Positions
        self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
        self.drop_joints = {
            State.RIGHT_TO_LEFT: [0.8464177250862122, -1.7617242972003382, -1.3163345495807093, -1.664525334035055, 1.5956381559371948, 0.03218962997198105],
            State.LEFT_TO_RIGHT: [-0.4250834623919886, -1.76178485551943, -1.3162863890277308, -1.6644414106952112, 1.5955902338027954, 0.03218962997198105],
            }
        self.workspaces = {
            State.RIGHT_TO_LEFT: [-0.610, -0.335, 0.140, 0.505, 0, 1],
            State.LEFT_TO_RIGHT: [-0.580, -0.305, -0.520, -0.160, 0, 1],
            }

        self.move_home_robot_state = self.get_robot_state(self.move_home_joints)

        self.dont_display_plan = True
        self.choose_random = True

        # Initializations
        self.state = State.BOOTUP
        self.agile_state = AgileState.WAIT_FOR_ONE

        self.gripper_data = 0
        self.agile_data = 0


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

        # Publisher for grasp arrows
        self.pose_publisher = rospy.Publisher("/pose_viz", PoseArray, queue_size=1)
        
        # Gripper nodes
        self.gripper_sub = rospy.Subscriber('/Robotiq2FGripperRobotInput', inputMsg.Robotiq2FGripper_robot_input, self.gripper_state_callback)
        self.gripper_pub = rospy.Publisher('/Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size=1)

        # Nodes for stitched point cloud
        self.generate_pcl = rospy.ServiceProxy('generate_pcl', PCLStitch)
        self.PCL_stitched_publisher = rospy.Publisher("/processed_PCL2_stitched", PointCloud2, queue_size=1)

        # Agile grasp node
        rospy.Subscriber("/detect_grasps/grasps", GraspListMsg, self.agile_callback)

    def agile_callback(self, data):
        self.agile_data = data
        self.agile_state = AGILE_STATE_TRANSITION[self.agile_state]


    def find_best_grasp(self, data, choose_random=False):
        offset_dist = 0.1
        max_angle = 90
        final_grasp_pose = 0
        final_grasp_pose_offset = 0

        num_bad_angle = 0
        num_bad_plan = 0

        poses = []
        if choose_random:
            #Shuffle grasps
            random.shuffle(data.grasps)
            rospy.loginfo('Grasps shuffled!')
        else:
            # Sort grasps by quality
            data.grasps.sort(key=lambda x : x.score, reverse=True)
            rospy.loginfo("Grasps Sorted!")
        
        # loop through grasps from high to low quality
        for g in data.grasps:
            if rospy.is_shutdown():
                break
            
            # Get grasp pose from agile grasp outputs
            R = np.zeros((3,3))
            R[:, 0] = vector3ToNumpy(g.approach)
            R[:, 1] = vector3ToNumpy(g.axis)
            R[:, 2] = np.cross(vector3ToNumpy(g.approach), vector3ToNumpy(g.axis))

            q = Quaternion(matrix=R)
            position =  g.surface
            rospy.loginfo("Grasp cam orientation found!")

            #Create poses for grasp and pulled back (offset) grasp
            p_base = PoseStamped()


            p_base.pose.position.x = position.x 
            p_base.pose.position.y = position.y 
            p_base.pose.position.z = position.z 

            p_base.pose.orientation.x = q[1]
            p_base.pose.orientation.y = q[2]
            p_base.pose.orientation.z = q[3]
            p_base.pose.orientation.w = q[0]

            p_base_offset = copy.deepcopy(p_base)
            p_base_offset.pose.position.x -= g.approach.x *offset_dist
            p_base_offset.pose.position.y -= g.approach.y *offset_dist
            p_base_offset.pose.position.z -= g.approach.z *offset_dist


            # Here we need to define the frame the pose is in for moveit
            p_base.header.frame_id = "base_link"
            p_base_offset.header.frame_id = "base_link"

            # Used for visualization
            poses.append(copy.deepcopy(p_base.pose))

            # Find angle between -z axis and approach
            approach_base = np.array([g.approach.x, g.approach.y, g.approach.z])
            approach_base = approach_base / np.linalg.norm(approach_base)
            theta_approach = np.arccos(np.dot(approach_base, np.array([0,0,-1])))*180/np.pi

            rospy.loginfo("Grasp base orientation found")  

            # If approach points up, no good            
            if theta_approach < max_angle:
                
                # Check if plan to grasp is valid
                self.move_group.set_start_state(self.move_home_robot_state)
                self.move_group.set_pose_target(p_base)
                plan_to_final = self.move_group.plan()
                self.move_group.clear_pose_targets()

                if plan_to_final.joint_trajectory.points:

                    #Check id plan to offset grasp pose is valid
                    self.move_group.set_start_state(self.move_home_robot_state)
                    self.move_group.set_pose_target(p_base_offset)
                    plan_offset = self.move_group.plan()
                    self.move_group.clear_pose_targets()

                    if plan_offset.joint_trajectory.points:
                        # If so, we've found the grasp to use
                        final_grasp_pose = p_base
                        final_grasp_pose_offset = p_base_offset
                        rospy.loginfo("Final grasp found!")
                        rospy.loginfo(" Angle: %.4f",  theta_approach)
                        # Only display the grasp being used
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

        # Publish grasp pose arrows
        posearray = PoseArray()
        posearray.poses = poses
        posearray.header.frame_id = "base_link"
        self.pose_publisher.publish(posearray)

        #print("final_grasp_pose", final_grasp_pose)
        rospy.loginfo("# bad angle: " + str(num_bad_angle))
        rospy.loginfo("# bad plan: " + str(num_bad_plan))

        if not final_grasp_pose:
            plan_offset = 0
            rospy.loginfo("No valid grasp found!")

        return final_grasp_pose_offset, plan_offset, final_grasp_pose

    # Use class variables to move to a pose
    def move_to_position(self, grasp_pose, plan=None):
        move_ur5(self.move_group, self.robot, self.display_trajectory_publisher, grasp_pose, plan, no_confirm=self.dont_display_plan)

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
        drop_joints = self.drop_joints[state]#### TODO: Define this dict
        dropped_flag = False

        #Move home
        self.move_group.set_start_state_to_current_state()
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.2)

        #Grab object
        self.move_to_position(final_grasp_pose_offset, plan_offset)
        rospy.sleep(0.2)
        self.move_to_position(final_grasp_pose)
        rospy.sleep(0.2)
        self.command_gripper(close_gripper_msg())
        rospy.sleep(1)

        #Move to drop position, checking if the object is dropped
        self.move_to_position(self.lift_up_pose())
        rospy.sleep(0.2)

        joints_to_move_to = [self.move_home_joints, drop_joints]
        for joints in joints_to_move_to:
            rospy.sleep(0.5)
            if self.gripper_data.gOBJ == 3:
                rospy.loginfo("Robot has missed/dropped object!")
                dropped_flag = True
                return True
            self.move_to_joint_position(joints)
        
        #Drop object
        self.command_gripper(open_gripper_msg())
        rospy.sleep(0.5)
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.2)

        return dropped_flag

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

    def main(self):
        rate = rospy.Rate(1)
        # Startup

        while not self.gripper_data and not rospy.is_shutdown():
            rospy.loginfo("Waiting for gripper to connect")
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
        rospy.sleep(0.1)
        rospy.loginfo("Moved to Home Position")

        #### TODO: Init number of object in each box (maybe from a ros param)
        # 
 
        ####TODO: Generate an intial box to grab from based on # of objects in the box
        self.state = State.RIGHT_TO_LEFT

        ####TODO: Set initial workspace based on current state
        ws_curr = self.workspaces[self.state]

        ####TODO: Init counter of failed grasps
        # failed_grasps = 0 <maybe?>

        while not rospy.is_shutdown():

            rospy.set_param("/detect_grasps/workspace", ws_curr)
            # Generate a point cloud from several readings
            self.agile_state = AgileState.WAIT_FOR_ONE
            rospy.loginfo("Generating point cloud")
            point_cloud = self.generate_pcl(int(self.state)) #### TODO: set generate_pcl input based on state
            self.PCL_stitched_publisher.publish(point_cloud.cloud)
            rospy.loginfo("Point cloud generated")

            #Wait for a valid reading from agile grasp
            while not rospy.is_shutdown() and self.agile_state is not AgileState.READY:
                rospy.loginfo("Waiting for agile grasp")
                self.command_gripper(close_gripper_msg())
                rospy.sleep(0.2)
                self.command_gripper(open_gripper_msg())
                rospy.sleep(2)
            
            rospy.loginfo("Grasp pose detection complete")

            ####TODO: sample from list randomly instead maybe?
            #Find best grasp from reading
            rospy.loginfo("Finding valid grasp")
            final_grasp_pose_offset, plan_offset, final_grasp_pose = self.find_best_grasp(self.agile_data, self.choose_random)
            
            drop_flag = None
            if final_grasp_pose:
                rospy.loginfo("Grasp found! Executing grasp")
                #Run the current motion on it 
                drop_flag = self.run_motion(self.state, final_grasp_pose_offset, plan_offset, final_grasp_pose)
            else:
                rospy.loginfo("No pose target generated!")

            rospy.loginfo("Switching state!")
            self.state = STATE_TRANSITION[self.state]
            ws_curr = self.workspaces[self.state]

            rospy.loginfo("Moving home")
            self.move_to_joint_position(self.move_home_joints)
            self.command_gripper(open_gripper_msg())
            rospy.sleep(.1)
            
            rate.sleep()


            ####TODO: 
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


if __name__ == '__main__':
    try:
        grasper = GraspExecutor()
        grasper.main()
    except KeyboardInterrupt:
        pass
