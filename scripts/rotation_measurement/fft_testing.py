import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq, irfft
import glob
import pdb
import matplotlib.pyplot as plt

input_path = "./data_processing/new_cylinder_data_csvs/"
output_path = "./test_kalman/"


avg_error_for_cutoffs = []
cutoff_list = []
# for cutoff in range(2, 20, 1):
for cutoff in range(1):
    cutoff_list.append(cutoff)
    for file in glob.glob(input_path + "*.csv"):
        avg_angle_error = []
        fig = plt.figure()

        data = pd.read_csv(file)
        # print(data)

        angles = data.to_numpy()[:,-2]
        angles -= angles[0]

        SAMPLE_RATE = 60
        DURATION = (len(angles) - 1) / SAMPLE_RATE
        N = len(angles) - 1

        velocity = []

        initial_iter = True
        for th in angles:
            if initial_iter:
                prev_angle = th
                initial_iter = False
            else:
                velocity.append((th - prev_angle)/(1/60))
                prev_angle = th

        yf = rfft(velocity)
        xf = rfftfreq(N, 1 / SAMPLE_RATE)
        # ax1  = fig.add_subplot(315)
        # ax1.title.set_text("Velocity FFT")
        # plt.plot(xf, np.abs(yf))
        # plt.show()

        # cutoff = len(yf) // 3
        cutoff = 11
        yf[cutoff:] = 0
        lpf_velocity = irfft(yf)

        angle_error = []
        prev_lpf_vel = 0
        integrated_angle = 0
        int_angle_list = [0]
        for i, v in enumerate(lpf_velocity):
            integrated_angle += (v - prev_lpf_vel) / 60
            int_angle_list.append(integrated_angle)
            true_angle = angles[i+1]
            angle_error.append(abs(true_angle - integrated_angle))

        avg_error = np.array(angle_error).mean()
        avg_angle_error.append(avg_error)
        print("Average angle error: ", avg_error)

        ax5  = fig.add_subplot(311)
        ax5.title.set_text("Angle")
        plt.plot(angles)
        plt.plot(int_angle_list)

        ax3  = fig.add_subplot(312)
        ax3.title.set_text("Velocity")
        plt.plot(velocity)
        plt.plot(lpf_velocity)

        ax2  = fig.add_subplot(313)
        ax2.title.set_text("Angle error")
        plt.plot(angle_error)
    
        fig.tight_layout()
        plt.show()

        # pdb.set_trace()

    total_avg_error = np.array(avg_angle_error).mean()
    print("Total average angle error: ", total_avg_error)
    avg_error_for_cutoffs.append(total_avg_error)

min_idx = np.argmin(avg_error_for_cutoffs)

print("\n\nLowest avg error: %.4f\nBest cutoff value: %i"%(avg_error_for_cutoffs[min_idx], cutoff_list[min_idx]))

plt.plot(cutoff_list, avg_error_for_cutoffs)
plt.show()