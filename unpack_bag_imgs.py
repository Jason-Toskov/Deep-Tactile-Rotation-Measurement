# From: https://idorobotics.com/2021/03/08/extracting-ros-bag-files-to-python/

import subprocess
import yaml
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np


FILENAME = 'Indoor'
ROOT_DIR = '/home/Dataset'
BAGFILE = 'data_1642136892.bag'

if __name__ == '__main__':
    bag = rosbag.Bag(BAGFILE)



    TOPIC = 'image'
    DESCRIPTION = 'color_'
    image_topic = bag.read_messages(TOPIC)
    for k, b in enumerate(image_topic):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(b.message, desired_encoding='bgr8') #b.message.encoding)
        #cv_image.astype(np.uint8)
        cv2.imwrite('./colour_imgs/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
        print('saved: ' + DESCRIPTION + str(b.timestamp) + '.png')
        print(k)



    bag.close()

    print('PROCESS COMPLETE')