#!/usr/bin/env python
import rospy

from sensor_msgs.msg import Image

class DataBagger:
    def __init__(self):
        rospy.init_node("Data_bagger", anonymous=True)

        self.image_topic = "/realsense/rgb"
        self.current_image = 0
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber("")  ##TODO
        ##TODO: Node to read tactile data

        print("Init!")

    def image_callback(self, data):
        self.current_image = data

    def main(self):
        rate = rospy.Rate(20)

        while not (self.current_image or rospy.is_shutdown()):
            rospy.sleep(1)
            rospy.loginfo("Waiting for camera!")

        while not rospy.is_shutdown():

            if ##Start data collection:
                ##init a bag
                while ##Still rotating:
                    #Get current time
                    #Save an image
                    #save forces
                    #sleep for rate
                ##once done, save bag
            
            
            rate.sleep()


if __name__ == "__main__":
    try:
        data_class = DataBagger()
        data_class.main()
    except KeyboardInterrupt:
        pass