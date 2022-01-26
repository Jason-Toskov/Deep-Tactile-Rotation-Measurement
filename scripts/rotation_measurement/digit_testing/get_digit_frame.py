from digit_interface import Digit, DigitHandler
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

digits = DigitHandler.list_digits()

past_frames = Queue(maxsize = 5)

print("Supported streams: \n {}".format(Digit.STREAMS))

d = Digit(digits[0]['serial']) # Unique serial number
d.connect()
d.set_resolution(Digit.STREAMS["QVGA"])
d.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

time_now = time.time()
frame1 = d.get_frame()
count=0
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
    cv2.imshow('fig1',frame2)
    # frame1 = copy.deepcopy(frame2 - frame1)
    cv2.waitKey(1)
    time_later = time.time()
    # print("fps: %f"%(1/(time_later-time_now)))
    # print()
    time_now = copy.deepcopy(time_later)

# d.show_view()
# d.disconnect()