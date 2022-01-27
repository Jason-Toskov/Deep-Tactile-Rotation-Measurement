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

FILE_DIR = './'
FOLDER_NAME = 'digitdata'
BAG_DIR = 'recorded_data_bags/'
OUTPUT_DIR = 'digit_data_unpacked/'

def digit_data_to_folder(folder, bridge, time_data, image_data, digit_data_0, digit_data_1, track_angle_srv):
    df = pd.DataFrame()
    df['true_angle'] = None
    df['timestep'] = None

    for i, (time, im, digit_0, digit_1) in enumerate(itertools.izip(time_data, image_data, digit_data_0, digit_data_1)):
        new_row = pd.Series(dtype='int64')
        response = track_angle_srv(im)
        new_row['true_angle'] = response.angle
        new_row['timestep'] = time.to_nsec()
        df = df.append(copy.deepcopy(new_row), ignore_index=True)

        dig_im_0 = bridge.imgmsg_to_cv2(digit_0, desired_encoding='BGR8')
        dig_im_1 = bridge.imgmsg_to_cv2(digit_1, desired_encoding='BGR8')

        cv2.imwrite(folder+"/"+str(i)+"_digit_0.jpeg", dig_im_0)
        cv2.imwrite(folder+"/"+str(i)+"_digit_1.jpeg", dig_im_1)

    df.to_csv(folder+"/"+"ground_truth.csv", index=False)


def main(track_angle_srv, reset_angle_srv):
    bridge = CvBridge()
    num_data_points = len(glob.glob(FILE_DIR+OUTPUT_DIR+'*/'))
    bag_list = glob.glob(FILE_DIR+BAG_DIR+'*.bag')
    print(FILE_DIR+BAG_DIR+'*.bag', bag_list)

    for bag_dir in bag_list:
        bag = rosbag.Bag(bag_dir)

        time_data = [msg.stamp for _, msg, _ in bag.read_messages(topics=['time'])]
        image_data = [msg for _, msg, _ in bag.read_messages(topics=['image'])]
        digit_data_0 = [msg for _, msg, _ in bag.read_messages(topics=['digit_0'])]
        digit_data_1 = [msg for _, msg, _ in bag.read_messages(topics=['digit_1'])]

        meta = [msg for _, msg, _ in bag.read_messages(topics=['metadata'])][0]

        if len(time_data) == len(image_data) == len(digit_data_0) == len(digit_data_1):
            #Folder naming convention is: <name>_<number of df>_<twist>_<gripperTwist>_<eeGroundRot>_<eeAirRot>_<gripperWidth>
            folder = [FILE_DIR,OUTPUT_DIR,FOLDER_NAME,'_',num_data_points,'_',meta.gripperTwist,'_',meta.eeGroundRot,'_',meta.eeAirRot,'_',meta.gripperWidth]
            folder = ''.join(list(map(str, folder)))
            os.mkdir(folder)

            digit_data_to_folder(folder, bridge, time_data, image_data, digit_data_0, digit_data_1, track_angle_srv)
            
            num_data_points += 1
        else:
            print('ERROR: bag ' + bag_dir +' had mismatched data!')

        reset_angle_srv()

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        # Create a new directory because it does not exist 
        os.makedirs(OUTPUT_DIR)
        print("The new directory is created!")

    rospy.wait_for_service('track_angle')
    try:
        track_angle_srv = rospy.ServiceProxy('track_angle', AngleTrack)
        rospy.loginfo("Angle tracking service available!")
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

    rospy.wait_for_service('reset_angle_tracking')
    try:
        reset_angle_srv = rospy.ServiceProxy('reset_angle_tracking', Empty)
        rospy.loginfo("Reset service available!")
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
    

    main(track_angle_srv, reset_angle_srv)