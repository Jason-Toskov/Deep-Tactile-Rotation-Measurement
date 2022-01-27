#!/usr/bin/env python
import rospy
import rosbag, copy
import glob
import itertools
import pandas as pd
from grasp_executor.srv import AngleTrack
from std_srvs.srv import Empty
import os
import pdb

from grasp_executor.msg import DataCollectState

FILE_DIR = './'
CSV_NAME = 'data'
BAG_DIR = 'recorded_data_bags/'
OUTPUT_DIR = 'recorded_data_csvs/'

def init_df(df):
    cols_sensor = ['gfX', 'gfY', 'gfZ', 'gtX', 'gtY', 'gtZ', 'friction_est', 'target_grip_force']
    cols_pillar = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ', 'in_contact']
    for i in range(2):
        for c in cols_sensor:
            df[c+'_sensor_'+str(i)] = None

        for j in range(9):
            for c in cols_pillar:
                df[c+'_pillar_'+str(j)+'_sensor_'+str(i)] = None
    
    df['true_angle'] = None
    df['timestep'] = None
    
    return df

def tactile_data_to_df(df, time_data, image_data, tactile_data_0, tactile_data_1, track_angle_srv): 
    cols_sensor = ['gfX', 'gfY', 'gfZ', 'gtX', 'gtY', 'gtZ', 'friction_est', 'target_grip_force']
    cols_pillar = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ', 'in_contact']

    for time, im, tac_0, tac_1 in itertools.izip(time_data, image_data, tactile_data_0, tactile_data_1):
        new_row = pd.Series(dtype='int64')
        print(type(im))
        response = track_angle_srv(im) #TODO: Call the blob_detector service here to get the angle
        new_row['true_angle'] = response.angle
        new_row['timestep'] = time.to_nsec()
        print(response.angle)

        for i in range(2):
            tac = tac_0 if i==0 else tac_1
            for attr in cols_sensor:
                new_row[attr+'_sensor_'+str(i)] = getattr(tac, attr)
            
            # pdb.set_trace()
            for j in range(9):
                for attr in cols_pillar:
                    new_row[attr+'_pillar_'+str(j)+'_sensor_'+str(i)] = getattr(tac.pillars[j], attr)
                    # print(j, tac.pillars[j], attr)
                    # raw_input()

        df = df.append(copy.deepcopy(new_row), ignore_index=True)
        # print(new_row)
        # raw_input()


    return df

def main(track_angle_srv, reset_angle_srv):
    num_df = len(glob.glob(FILE_DIR+OUTPUT_DIR+'*.csv'))

    bag_list = glob.glob(FILE_DIR+BAG_DIR+'*.bag')
    print(FILE_DIR+BAG_DIR+'*.bag', bag_list)

    for bag_dir in bag_list:
        bag = rosbag.Bag(bag_dir)
        df = pd.DataFrame() 
        df = init_df(df)


        time_data = [msg.stamp for _, msg, _ in bag.read_messages(topics=['time'])]
        image_data = [msg for _, msg, _ in bag.read_messages(topics=['image'])]
        tactile_data_0 = [msg for _, msg, _ in bag.read_messages(topics=['tactile_0'])]
        tactile_data_1 = [msg for _, msg, _ in bag.read_messages(topics=['tactile_1'])]

        meta = [msg for _, msg, _ in bag.read_messages(topics=['metadata'])][0]

        timestep_0 = [msg.tus for msg in tactile_data_0]
        timestep_1 = [msg.tus for msg in tactile_data_0]

        print(len(timestep_0))
        print(len(set(timestep_0)))
        print(len(timestep_1))
        print(len(set(timestep_1)))

        # pdb.set_trace()

        if len(time_data) == len(image_data) == len(tactile_data_0) == len(tactile_data_1):
            df = tactile_data_to_df(df, time_data, image_data, tactile_data_0, tactile_data_1, track_angle_srv)

            #Naming convention is: <name>_<number of df>_<twist>_<gripperTwist>_<eeGroundRot>_<eeAirRot>_<gripperWidth>.csv
            data = [FILE_DIR,OUTPUT_DIR,CSV_NAME,'_',num_df,'_',meta.gripperTwist,'_',meta.eeGroundRot,'_',meta.eeAirRot,'_',meta.gripperWidth,'.csv']
            df.to_csv(''.join(list(map(str, data))), index=False)
            num_df += 1
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