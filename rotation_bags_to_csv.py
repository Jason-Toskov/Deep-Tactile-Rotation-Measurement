import rosbag
import glob
import itertools
import pandas as pd

FILE_DIR = './'
CSV_NAME = 'data'
BAG_DIR = 'data_bags/'
OUTPUT_DIR = 'data_csvs/'

def init_df(df):
    cols_sensor = ['gfX', 'gfY', 'gfZ', 'gtX', 'gtY', 'gtZ', 'friction_est', 'target_grip_force']
    cols_pillar = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ', 'in_contact']
    for i in range(2):
        for c in cols_sensor:
            df[c+'_sensor_'+str(i)] = None

        for j in range(9):
            for c in cols_pillar:
                df[c+'_pillar_'+str(j)+'_sensor_'+str(i)] = None
    
    return df

def tactile_data_to_df(df, image_data, tactile_data_0, tactile_data_1): 
    cols_sensor = ['gfX', 'gfY', 'gfZ', 'gtX', 'gtY', 'gtZ', 'friction_est', 'target_grip_force']
    cols_pillar = ['dX', 'dY', 'dZ', 'fX', 'fY', 'fZ', 'in_contact']

    for im, tac_0, tac_1 in itertools.izip(image_data, tactile_data_0, tactile_data_1):
        new_row = pd.Series(dtype='int64')
        angle = angle_from_image(im) #TODO: Call the blob_detector service here to get the angle
        new_row['true_angle'] = angle

        for i in range(2):
            tac = tac_0 if i==0 else tac_1
            for attr in cols_sensor:
                new_row[attr+'_sensor_'+str(i)] = getattr(tac, attr)
            
            for j in range(9):
                for attr in cols_pillar:
                    df[attr+'_pillar_'+str(j)+'_sensor_'+str(i)] = getattr(tac.pillars[j], attr)

        df.append(new_row, ignore_index=True)

    return df

def main():
    # df = pd.DataFrame() 
    # df = init_df(df, num_sensors=NUM_SENSORS)
    num_df = len(glob.glob(FILE_DIR+OUTPUT_DIR+'*.csv'))

    bag_list = glob.glob(FILE_DIR+BAG_DIR+'*.bag')

    for bag_dir in bag_list:
        bag = rosbag.Bag(bag_dir)
        df = pd.DataFrame() 
        df = init_df(df)

        image_data = [msg for _, msg, _ in bag.read_message(topics=['image'])]
        tactile_data_0 = [msg for _, msg, _ in bag.read_message(topics=['tactile_0'])]
        tactile_data_1 = [msg for _, msg, _ in bag.read_message(topics=['tactile_1'])]

        if len(image_data) == len(tactile_data_0) == len(tactile_data_1):
            df = tactile_data_to_df(df, image_data, tactile_data_0, tactile_data_1)
            df.to_csv(FILE_DIR+OUTPUT_DIR+CSV_NAME+'_'+str(num_df)+'.csv', index=False)
            num_df += 1
        else:
            print('ERROR: bag ' + bag_dir +' had mismatched data!')

        
if __name__ == "__main__":
    main()