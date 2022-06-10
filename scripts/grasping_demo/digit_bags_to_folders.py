#!/usr/bin/env python
import rospy
import rosbag
import glob
from grasp_executor.srv import AngleTrack
from std_srvs.srv import Empty
import os
import pandas as pd
import itertools
from cv_bridge import CvBridge, CvBridgeError
import copy
import cv2
from rosbag.bag import ROSBagUnindexedException

import numpy as np
import open3d as o3d

FILE_DIR = './'
FOLDER_NAME = 'grasp_data_bags'
BAG_DIR = 'grasp_data_bags/'
OUTPUT_DIR = 'both_data_unpacked/'

def main():
    bridge = CvBridge()
    num_data_points = len(glob.glob(FILE_DIR+OUTPUT_DIR+'*/'))
    bag_list = glob.glob(FILE_DIR+BAG_DIR+'*.bag')
    print(FILE_DIR+BAG_DIR+'*.bag', bag_list)

    for i, bag_dir in enumerate(bag_list):
        try:
            bag = rosbag.Bag(bag_dir)
        except ROSBagUnindexedException:
            print('Unindexed bag'+bag_dir+' with i='+str(i)+' (total list length = '+str(num_data_points)+')')

        time_data = [msg.stamp for _, msg, _ in bag.read_messages(topics=['time'])]
        object_id = [msg for _, msg, _ in bag.read_messages(topics=['object_id'])]
        rgb_image = [msg for _, msg, _ in bag.read_messages(topics=['rgb_image'])]
        depth_image = [msg for _, msg, _ in bag.read_messages(topics=['depth_image'])]
        point_cloud = [msg for _, msg, _ in bag.read_messages(topics=['point_cloud'])]
        trans = [msg for _, msg, _ in bag.read_messages(topics=['trans'])]
        rot = [msg for _, msg, _ in bag.read_messages(topics=['rot'])]
        cam_info = [msg for _, msg, _ in bag.read_messages(topics=['cam_info'])]
        ee_pose = [msg for _, msg, _ in bag.read_messages(topics=['ee_pose'])]
        robot_pose = [msg for _, msg, _ in bag.read_messages(topics=['robot_pose'])]
        success = [msg for _, msg, _ in bag.read_messages(topics=['success'])]
        stable_success = [msg for _, msg, _ in bag.read_messages(topics=['stable_success'])]

        print(len(time_data),len(object_id),len(rgb_image),len(depth_image),len(point_cloud),len(trans),len(rot),len(cam_info),len(robot_pose), len(ee_pose), len(success))
        
        if len(time_data) == len(object_id) == len(rgb_image) == len(depth_image) == len(point_cloud) == len(trans) == len(rot) == len(cam_info) == len(robot_pose) == len(ee_pose) == len(success):
            # #Folder naming convention is: <name>_<number of df>_<twist>_<gripperTwist>_<eeGroundRot>_<eeAirRot>_<gripperWidth>
            # folder = [FILE_DIR,OUTPUT_DIR,FOLDER_NAME,'_',num_data_points,'_',meta.gripperTwist,'_',meta.eeGroundRot,'_',meta.eeAirRot,'_',meta.gripperWidth]
            # folder = ''.join(list(map(str, folder)))
            # os.mkdir(folder)

            # digit_data_to_folder(folder, bridge, time_data, image_data, digit_data_0, digit_data_1, track_angle_srv)
            for i, (time, object_id, rgb, depth, point_cloud, trans, rot, cam_info, ee, robot, success, stable_success) in enumerate(itertools.izip(time_data, object_id, rgb_image, depth_image, point_cloud, trans, rot, cam_info, ee_pose, robot_pose, success, stable_success)):
            # num_data_points += 1
                cv2_rgb = bridge.imgmsg_to_cv2(rgb, desired_encoding="bgr8")
                cv2_depth = bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
                cv2.imshow("rgb", cv2_rgb)
                cv2.imshow("depth", cv2_depth)
                print("object_id: ", object_id)
                print("trans: ", trans)
                print("rot: ", rot)
                print("cam_info: ", cam_info)
                print("ee_pose: ", ee)
                print("robot_pose: ", robot)
                print("success: ", success)
                print("stable success: ", stable_success)
                cv2.waitKey(10000)
        else:
            print('ERROR: bag ' + bag_dir +' had mismatched data!')

if __name__ == "__main__":
    if not os.path.exists(FILE_DIR + OUTPUT_DIR):
        # Create a new directory because it does not exist 
        os.makedirs(FILE_DIR + OUTPUT_DIR)
        print("The new directory is created!")

    main()