#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

import os #for time stamp
import calendar #for time stamp
import time #for time stamp

from scipy.spatial import KDTree#added for KDtree same as waypoint_updater
# added for KDtree part in looking forward
import numpy as np

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.light_classifier = None # Init here because TLClassifier constructor takes a long time

        # set to True, if you want to use the simulator traffic light color (only for testing)
        self.USE_SIMULATOR_TRAFFIC_LIGHT_COLOR = False # True # False # set to true 28 July as part of new dataset creation
        self.RECORD_CAMERA_IMAGES = False # True  # set to True to record output from camera works idependently from above Boolean
        
        #from waypoint_updater to make similar KDtree
        self.waypoints_2d = None
        self.waypoint_tree = None
        # John two variables added to sample frame rate every so many frames to help reduce lag when cmera on
        # thgis experiemnt did not work well
        self.count = 0
        self.frame_sample_rate = 1 #counting from 0 3 will give one in 4 frames from camera stream

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=2**24)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        #john adding site or sim boolean as suggested in section 9 of classroom
        self.is_site = self.config['is_site'] # this gets sent to the classifier below

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.is_site)#adding is_site boolean from above
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
        #another expeirment moved from below to allow camera to be turned off
        # this works the cars slows and stops fine at the traffic lights without processing the images in the image callback
        # this is ON for testing OFF for final version
        if self.USE_SIMULATOR_TRAFFIC_LIGHT_COLOR == True :
            #this is the same coe from image_cb but here for when we want to follow waypoints and stop at lights without camera son and using simulator provided traffic light states
            light_wp, state = self.process_traffic_lights()
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1
        


    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        #next lines from waypoint_updater for similar KDtree
        if not self.waypoints_2d:
            self.waypoints_2d = [ [ waypoint.pose.pose.position.x , waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints ]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        #experiment to avoid image_cb...failed
        #light_wp, state = self.process_traffic_lights()



    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        #added by John to sample only every X images ....failed experiment going back to throttle limit to slow car
        #if self.count < self.frame_sample_rate :
        #    self.count += 1
        #    return #less than the count increment and return ship the rest of the function
        #elif self.count == self.frame_sample_rate :
        #    self.count = 0 #reset the counter and continue
        #END of JOHN's addition
        #############
        #John is adding a part here to deal with the error in the second submission, which was some errors thrown after the image_cb was woprking but before the tl_classifier was initialiasted. In the site this was several errors that seemed to have little effect but we want to deal with them anyway
        
        if not self.light_classifier :
            return
        ##end of extra return section ....John
        
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
         #added 31 august if statement because when USE_SIMULATOR_TRAFFIC_ True waypoint are updated in pose
        if self.USE_SIMULATOR_TRAFFIC_LIGHT_COLOR == False :
            
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))           

                rospy.loginfo('In tl_detector elif section light_wp:{}'.format(Int32(self.last_wp) ) )
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                rospy.loginfo('In tl_detector else section last_wp:{}'.format( Int32(self.last_wp) ) )
            self.state_count += 1
        
        #maybe put call to generate test output here? 31 Aug
        if self.RECORD_CAMERA_IMAGES == True : #this was in process image moved 31 aug
            self.create_training_data(state) # 
        
        

    def get_closest_waypoint(self, x,y):# original was ... pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        #todo more... I think this needs more to make it closest waypoint AHEAD of car as there is a possibility if returnsing the traffic lights behind the car.....on second thoughts that might not be needed....made no difference to "site"

        closest_idx =  self.waypoint_tree.query([x,y], 1 )[1] # was ([x,y], 1 )[1] see below
#
#
#2 july maybe this ahead part is breaking it?? will remove as tesst#

        # I am going to copy this into tl_detector which is already slecting nearest waypoint but not nessecarily ahead waypoint might be causeing problems...John
        '''
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx -  1]

        #Equation of hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect - prev_vect , pos_vect - cl_vect )

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        '''
        return closest_idx #0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #For testing just return the light state
        #return light.state #swtiching to our classifier
        #original below for hen using classifier

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)# Joosts is different name ...I copied it into our
        # returning this to something that works

   #experiment to create trining images
#this works and will be used if we want full screen images from the simulator
    def create_training_data(self, state):
        f_name = "sim_tl_{}_{}.jpg".format(calendar.timegm(time.gmtime()), state) # self.light_label(state))
        #dir = './data/train/sim' #added to 28 July to put data into color folders
        dir_RED = './data/RED'
        dir_GREEN = './data/GREEN'
        dir_YELLOW = './data/YELLOW'
        dir_UNKNOWN = './data/UNKNOWN'
        # making 4 dirs at same time if one is not there make them all
        #new 28 July
        if not os.path.exists(dir_RED):
            os.makedirs(dir_RED)
            os.makedirs(dir_GREEN)
            os.makedirs(dir_YELLOW)
            os.makedirs(dir_UNKNOWN)

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image)
        cv_image = cv_image[:, :, ::-1]
        #cv2.imwrite('{}/{}'.format(dir, f_name), cv_image)
        #added 28 july to spearate images into folders for dataset
        if state == 0 : #RED
            cv2.imwrite('{}/{}'.format(dir_RED, f_name), cv_image)
        elif state == 2 : #GREEN
            cv2.imwrite('{}/{}'.format(dir_GREEN, f_name), cv_image)
        elif state == 1 : # YELLOW
            cv2.imwrite('{}/{}'.format(dir_YELLOW, f_name), cv_image)
        elif state == 4 : #UNKNOWN
            cv2.imwrite('{}/{}'.format(dir_UNKNOWN, f_name), cv_image)#had unkMon here bug!
            
            


    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # from Walkthrough video
        closest_light = None
        line_wp_idx = None
        #light = None #not in video
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)# from Walkthrough
            #car_position = self.get_closest_waypoint(self.pose.pose) # not in Walkthrough video
        #TODO find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
        for i , light in enumerate(self.lights):
            # get stopline waypoint index
            line = stop_line_positions[i]
            temp_wp_idx = self.get_closest_waypoint(line[0],line[1])
            #Find the closest stopline index
            d = temp_wp_idx - car_wp_idx
            if d >= 0 and d < diff:
                diff = d
                closest_light = light
                line_wp_idx = temp_wp_idx

        rospy.loginfo('Process traffic lights...')

        if closest_light:
            if self.USE_SIMULATOR_TRAFFIC_LIGHT_COLOR:
                state = closest_light.state  # from simulator
                rospy.loginfo('State from simulator:{}'.format(state))
            else:
                state = self.get_light_state(closest_light) # from image
             #experiment to create test images from https://github.com/swap1712/carnd-capstone/blob/master/ros/src/tl_detector/tl_detector.py
            #this exerpiemnt to create traiing data works and might be used if we need it
            # in the end 5000+ images made for training by calling this function
            #moving below to image_cb
            #if self.RECORD_CAMERA_IMAGES == True : # self.has_image : throws error self.has_image may not be created yet
            #    self.create_training_data(state) # this works! # back on 28 July for more dataset

            return line_wp_idx, state
        return -1, TrafficLight.UNKNOWN
        # below Original and  not in Walkthrough video
        '''
        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN
        '''

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
