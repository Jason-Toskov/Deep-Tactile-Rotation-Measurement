from digit_interface import Digit, DigitHandler
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

digits = DigitHandler.list_digits()

past_frames = Queue(maxsize = 2)

print("Supported streams: \n {}".format(Digit.STREAMS))

d = Digit(digits[0]['serial']) # Unique serial number
d.connect()
d.set_resolution(Digit.STREAMS["QVGA"])
d.set_fps(Digit.STREAMS["QVGA"]["fps"]["60fps"])

time_now = time.time()
count=0
while True:
    frame = d.get_frame()
    past_frames.put(frame)

    if past_frames.full():
        comp_frame = past_frames.get()
        diff_im = cv2.absdiff(comp_frame, frame)
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
        cv2.imshow('fig1',frame)
        frame1 = copy.deepcopy(diff_im)
        cv2.waitKey(1)
        time_later = time.time()
        # print("fps: %f"%(1/(time_later-time_now)))
        # print()
        time_now = copy.deepcopy(time_later)

# d.show_view()
# d.disconnect()