from digit_interface import Digit, DigitHandler
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
import os

digits = DigitHandler.list_digits()

past_frames = Queue(maxsize = 5)

print("Supported streams: \n {}".format(Digit.STREAMS))

d = Digit(digits[0]['serial']) # Unique serial number
d.connect()

# d.set_resolution(Digit.STREAMS["VGA"])
# d.set_fps(Digit.STREAMS["VGA"]["fps"]["30fps"])


d.set_resolution(Digit.STREAMS["QVGA"])
d.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

folder_path = 'Digit_Video_fork6_1digits_QVGA_NEWSENSOR/'
if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

time_now = time.time()
frame1 = d.get_frame()
count=0
input("Press enter when ready:")
while True:
    frame2 = d.get_frame()
    # print(type(frame))
    diff_im = cv2.absdiff(frame2, frame1)
    diff_im[diff_im < 0] = 0
    diff_im =(diff_im.astype(float) * 255 / diff_im.max()).astype('uint8')

    if count < 300:
        count += 1
    else:
        # plt.hist((diff_im).flatten(), bins='auto')
        # plt.show()
        pass

    # breakpoint(diff_im)
    
    # print(frame2)
    # print(type(diff_im))
    # print(diff_im.max())
    # cv2.imshow('fig1',frame2)
    cv2.imwrite('./'+folder_path+'im_'+str(round(time_now))+'.png', frame2)
    # frame1 = copy.deepcopy(frame2 - frame1)
    # cv2.waitKey(1)
    time_later = time.time()
    print("fps: %f"%(1/(time_later-time_now)))
    print()
    time_now = copy.deepcopy(time_later)

# d.show_view()
# d.disconnect()