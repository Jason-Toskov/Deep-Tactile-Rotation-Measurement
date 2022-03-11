import glob, numpy as np
from unicodedata import decimal
from functools import reduce
output_folder = "twist_long_data_csvs"
data_folder = "long_data_csvs"

def flatten(l):
    # this is a cool way to flatten a list of lists lol
    return sum(l, [])

tactile_order = np.array([[0,1,2], [3,4,5], [6,7,8]])

for i in glob.glob(data_folder + "/*.csv"):

    for j in range(2):
        # to flip 90 degrees, we can transpose and flip the column ordering
        flip = np.array([[0,0,1], [0,1,0], [1,0,0]])
        # do a 180 degree rotation
        if j == 1:
            tactile_order = tactile_order.T @ flip
            tactile_order = tactile_order.T @ flip
        indices = tactile_order.flatten()
        new_file_name = i.replace(data_folder, output_folder)
        new_file_name = new_file_name.replace(".csv", f"_twist{j}.csv")

        with open(new_file_name, "w") as new_file:
        
            for line in open(i):

                sample = line.split(",")

                sensor0_global_data = sample[:8]
                runningIndex = 8
                sensor0_pillars = np.array_split(
                    sample[runningIndex:runningIndex+7*9], 9)

                runningIndex += 7*9
                sensor1_global_data = sample[runningIndex:8+runningIndex]
                runningIndex += 8

                sensor1_pillars = np.array_split(
                    sample[runningIndex:runningIndex+7*9], 9)

                runningIndex += 7*9
                final_data = sample[runningIndex:]

                sensor0_pillars_reorder = zip(
                    indices, sensor0_pillars)
                sensor0_pillars = sorted(sensor0_pillars_reorder, key=lambda x: x[0])
                sensor0_pillars = flatten(list(map(lambda x: list(x[1]), sensor0_pillars)))

                sensor1_pillars_reorder = zip(
                    indices, sensor1_pillars)
                sensor1_pillars = sorted(sensor1_pillars_reorder, key=lambda x: x[0])
                sensor1_pillars = flatten(list(map(lambda x: list(x[1]), sensor1_pillars)))

                if j == 0:
                    data = sensor0_global_data + sensor0_pillars + sensor1_global_data + sensor1_pillars + final_data
                else:
                    data = sensor1_global_data + sensor1_pillars + sensor0_global_data + sensor0_pillars + final_data

                new_file.write(",".join(data))