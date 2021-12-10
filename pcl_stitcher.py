#!/usr/bin/env python

import rospy
import sys
import moveit_commander
import moveit_msgs.msg
import copy
from tqdm import tqdm

import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import ros_numpy.point_cloud2 as rpc2
import pdb
import tf
from tf import TransformListener
from geometry_msgs.msg import PoseStamped



class PCLStitcher:
    def __init__(self):
        rospy.init_node('PCL_stitcher', anonymous=True)

        # Moveit
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        # TF
        self.pcl_rosmsg = 0
        self.tf_listener_ = TransformListener()
        self.tf_listener_.waitForTransform("/camera_link", "/base_link", rospy.Time(), rospy.Duration(4))
        self.pose_fwd = PoseStamped()
        self.pose_fwd.header.frame_id = 'camera_link'
        self.pose_inv = PoseStamped()
        self.pose_inv.header.frame_id = 'base_link'


        self.PCL_reader = rospy.Subscriber("/realsense/cloud", PointCloud2, self.cloud_callback)

        self.move_home_joints = [ 0.0030537303537130356,-1.5737221876727503, -1.4044225851642054, -1.7411778608905237, 1.6028796434402466, 0.03232145681977272]
        self.view_joints_1 = [0.24985386431217194, -0.702608887349264, -2.0076406637774866, -1.7586587111102503, 1.5221580266952515, 0.25777095556259155]
        self.view_joints_2 = [0.09033633768558502, -2.460919205342428, -0.5586937109576624, -2.818702522908346, 1.687928318977356, 0.03240497037768364]


    def cloud_callback(self, pcl):
        self.pcl_rosmsg = pcl

    def transformed_vector(self, x):
        x_transformed = np.matmul(self.transform_matrix, np.array([x[0], x[1], x[2], 1]))
        #pdb.set_trace()
        return x_transformed[:3]

    def check_box_bounds(self, x):
        x = self.transformed_vector(x)
        
        if -0.78 < x[0] < -0.41 :
            if -0.12 < x[1] < 0.158:
                return True
        return False

    def move_to_joint_position(self, joint_array):
        self.move_group.set_joint_value_target(joint_array)
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

    def transform_point_cloud(self, pcl, tf_matrix, new_pcl):
        pcl_temp = np.copy(pcl)
        pcl_temp_2 = np.array(pcl.tolist(), dtype='uint32')
        pcl_temp = pcl_temp.view((float, len(pcl_temp.dtype.names)))
        pcl_temp[:,3] = 1.0
        pcl_TF = np.matmul(tf_matrix, pcl_temp.T).T
        #pcl_transformed[:,3] = pcl_temp_2[:,3]
        pdb.set_trace()
        # pcl_list = list(zip(*pcl_transformed.T))
        # pcl_rec = np.recarray(pcl_list, dtype = pcl.dtype)

        blank_array = np.copy(new_pcl)[:0]
        for i in tqdm(range(len(pcl))):
            # This is VERY inefficient, and should not be used
            # There has to be a better way!!!!
            blank_array = np.append(blank_array, np.array([(pcl_TF[i][0], pcl_TF[i][1], pcl_TF[i][2], pcl_temp_2[i][3])], dtype=pcl.dtype))
        pdb.set_trace()
        new_pcl = np.append(new_pcl, blank_array)
        return new_pcl
        


    def main(self):
        rate = rospy.Rate(1)

        # Go to move home position using joint
        self.move_to_joint_position(self.move_home_joints)
        rospy.sleep(0.1)
        rospy.loginfo("Moved to Home Position")

        self.move_to_joint_position(self.view_joints_1)
        rospy.sleep(0.1)
        rospy.loginfo("Moved to first view position")
        rospy.sleep(2)
        TF_matrix_1 = self.tf_listener_.asMatrix("/base_link", self.pose_fwd.header)
        TF_matrix_1_inv = self.tf_listener_.asMatrix("/camera_link", self.pose_inv.header)
        pcl_1 = self.pcl_rosmsg
        pcl_1_np = rpc2.pointcloud2_to_array(pcl_1)

        new_pcl = np.copy(pcl_1_np)[:0]

        self.move_to_joint_position(self.view_joints_2)
        rospy.sleep(0.1)
        rospy.loginfo("Moved to second view position")
        rospy.sleep(2)
        TF_matrix_2 = self.tf_listener_.asMatrix("/base_link", self.pose_fwd.header)
        TF_matrix_2_inv = self.tf_listener_.asMatrix("/camera_link", self.pose_inv.header)
        pcl_2 = self.pcl_rosmsg
        pcl_2_np = rpc2.pointcloud2_to_array(pcl_2)

        
        new_pcl = self.transform_point_cloud(pcl_1_np, TF_matrix_1, new_pcl)

        pdb.set_trace()




if __name__ == '__main__':
    try:
        pcl_inst = PCLStitcher()
        pcl_inst.main()
    except KeyboardInterrupt:
        pass