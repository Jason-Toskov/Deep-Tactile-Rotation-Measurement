from black import out
from filterpy.kalman import KalmanFilter
import numpy as np, glob
import matplotlib.pyplot as plt

input_path = "./model_papilarray/augmented_long_data_csvs/"
output_path = "./model_papilarray/augmented_long_data_csvs_filtered/"

for file in glob.glob(input_path + "*.csv"):


    f = KalmanFilter (dim_x=2, dim_z=1)

    initial_state_set = False

    # transition matrix
    f.F = np.array([[1.,1/60],
                    [0.,1.]])

    # measurement function
    f.H = np.array([[1.,0.]])

    # covairance function
    f.P *= 0.00001
    # low measurement noise
    f.R = 0.00001

    velocity = []
    velocity_estimate = []
    velocity_estimate_from_kalman = []

    pos = []
    pos_estimate = []

    from filterpy.common import Q_discrete_white_noise
    f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    prev_angle = None
    output = []
    print(f"Starting file: {file}")
    for index,i in enumerate(open(file)):
        if index == 0:
            output.append(f"{i.strip()},kalman_pos, kalman_vel, base_vel, vel_from_kal_pos")
            continue

        z = float(i.split(",")[-2])
        if not initial_state_set:
            initial_state_set = True
            # initial state
            f.x = np.array([z, 0.])
            prev_angle = z
            prev_kalman_angle = f.x[0]
        else:
            f.predict()

            f.update([z])
            # print(f.x, z)

            velocity.append(f.x[1])        
            velocity_estimate.append((z - prev_angle) / (1/60))
            velocity_estimate_from_kalman.append((f.x[0] - prev_kalman_angle) / (1/60))

            pos.append(f.x[0])
            pos_estimate.append(z)

            

            # input()
            # output.append(f"{i.strip()},{f.x[0]},{f.x[1]}")
            temp_vel_est = (z - prev_angle) / (1/60)
            temp_vel_kal_est = (f.x[0] - prev_kalman_angle) / (1/60)
            
            output.append(f"{i.strip()},{f.x[0]},{f.x[1]},{temp_vel_est},{temp_vel_kal_est}") #store measured (noisy) velocity

            prev_angle = z
            prev_kalman_angle = f.x[0]

    x = open(output_path + 'kalman_' + file.split('/')[-1], "w")
    x.write("\n".join(output))
    x.close()

    plt.subplot(1, 2, 1)
    plt.plot(velocity, label="kalman filter")
    plt.plot(velocity_estimate, label="measured")
    plt.plot(velocity_estimate_from_kalman, label="derived from kalman pos")
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(pos, label="kalman filter")
    plt.plot(pos_estimate, label="measured")


    plt.legend()
    plt.savefig(output_path + 'kalman_' + file.split('/')[-1].split('.csv')[0] + ".png")
    plt.clf()
