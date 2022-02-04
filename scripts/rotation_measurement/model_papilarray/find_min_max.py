from lstm_papilarray import TactileDataset
import numpy as np
data = TactileDataset('./Data/', label_scale = 1)

# max_values, max_angles = data.getItem(0)
# min_values, min_angles = data.getItem(0)
# max_values = np.max(max_values.numpy(), axis=0)
# min_values = np.min(min_values.numpy(), axis=0)

# max_angles = np.max(max_angles)
# min_angles = np.min(min_angles)
max_values = None
min_values = None
# print(min_angles, max_angles)

for index in range(len(data)):
    x = data.getItem(index)
    # print(x.shape)
    
    if max_values is not None:
        tmp_max_values = np.vstack((max_values, x))
        max_values = np.max(tmp_max_values, axis=0)

        tmp_min_values = np.vstack((min_values, x))
        min_values = np.min(tmp_min_values, axis=0)

    else:
        max_values = np.max(x, axis=0)
        min_values = np.max(x, axis=0)
    
keep = np.where(min_values != max_values)
ignore = np.where(min_values == max_values)

print("keep", keep)
print("ignore", np.where(min_values == max_values))
np.save("keep_index", keep)
np.save("max_values", max_values[keep])
np.save("min_values", min_values[keep])
