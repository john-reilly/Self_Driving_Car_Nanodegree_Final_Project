import os
import time
import calendar

import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image, ImageOps
import numpy as np

import cv2

from object_classifier import ObjectClassifier
from color_classifier import ColorClassifier
#from traffic_light_colors import TrafficLight #had isseus around the emun import so useing styx.msg below
from styx_msgs.msg import TrafficLight
import rospy

class TLClassifier(object):
    #john  added boolean is_site to indicate site or sim classifer to load
    def __init__(self, is_site): 
        # init object classifier (step 1)
        rospy.loginfo('TLClassifier start inititialization')
        self.object_classifier = ObjectClassifier()

        # init traffic light color classifier (step 2)
        #john added is_site log output 
        if is_site:
            rospy.loginfo('is_site TRUE using site classifer')
        else :
            rospy.loginfo('is_site FALSE using sim classifer')
            
        self.color_classifier = ColorClassifier(is_site)#john ...now sending boolena to color classifier
        
        rospy.loginfo('TLClassifier inititialized')

        self.RECORD_CROPPED_IMAGES = False

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        rospy.loginfo('TLClassifier received image')

        # convert from bgr 2 rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # step 1
        traffic_light_images = self.object_classifier.get_traffic_light_images(image)
        
        traffic_light_color = self.color_classifier.predict_images(traffic_light_images)

        tf_color = ['RED', 'YELLOW', 'GREEN', 'UNDEFINED', 'UNKNOWN'] # Not so nice ;-( but want to show text
        rospy.loginfo('Traffic light detected {}'.format(tf_color[traffic_light_color]))

        if self.RECORD_CROPPED_IMAGES:
            dir = './data/cropped/'
            if not os.path.exists(dir):
                os.makedirs(dir)

            for idx, image in enumerate(traffic_light_images):
                f_name = "sim_tl_{}_{}_{}.jpg".format(calendar.timegm(time.gmtime()), tf_color[traffic_light_color], idx)
                image.save(dir + f_name)

        return traffic_light_color

