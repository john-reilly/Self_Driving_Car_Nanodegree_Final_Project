#!/usr/bin/env python
# the change includes parameters added line30-34, line49,line97-143
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32, Bool #forgot to add this to understand the imformationcoming into ie ints and bools

import math

#mine added
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 30# 10 made no imporvement in green dot issue 10 # 20 # Number of waypoints we will publish. You can change this number
STOP_LINE_MARGIN = 4
MAX_DECEL = 0.5
CONSTANT_DECEL = 1
#PUBLISHING_RATE = 20  # Rate (Hz) of waypoint publishing
#LOGGING_THROTTLE_FACTOR = PUBLISHING_RATE * 2  # Only log at this rate (1 / Hz)
class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        #by John : the value self.stopline_wp_idx was not getting updated therefore the condtions in generate_lane were never met to decelerate the waypoints.
        # the below subscrtiption to the traffic_light node should update the stopline_wp_idx and allow the decellerate to happend
        #also the fraffic call back needs to be adjusted to generate the call back for the subscription
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb) #from /ros/src/tl_detector/tl_detector.py

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        #OOOOMMMMGGGG was the / the problem!!!! adding / no difference
        #self.final_waypoints_pub        = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.final_waypoints_pub        = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        #  d other member variables you need below
        self.pose = None
        self.stopline_wp_idx = -1 # goes for a while and stops OK with 500 here 500 #-1 #experiment to hard setting to see where problem is which node...
        self.base_waypoints =  None
        self.waypoints_2d = None
        self.waypoint_tree = None
        #I am putting a init messag here....it does not seem to porduce and output
        rospy.loginfo('In __init__ waypoint' )
        
        self.loop()
        
    def loop(self):
        rate = rospy.Rate(50)# at 10 no obvious difference maybe a little better# 50 original value # again trying to deal with the probelm of leaving the green dots behind I am experiemnting with the refresh rate in DBW_node and here
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                #Get closest waypoint
                #below lineno longer needed ? John
                #closest_waypoint_idx = self.get_closest_waypoint_idx()#I had idx here error? put x back in as video had it
                self.publish_waypoints()# this causes error as we go to Complete Waypoint updater we don't need to pass closest waypoint here the generate_lane fiunction does this now (closest_waypoint_idx)
            rate.sleep()
        

       # rospy.spin() # no longer needed?
    def get_closest_waypoint_idx(self):#added x here video did not have it
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        #below if statement was early experiment prints not needed....
        #if self.waypoint_tree :# i want to see does it exist these lines seem to do nothing think I need special ros log output
        #    print("waypoint tree status: exists")  
        #if not self.waypoint_tree :# i want to see does it exist
        #    print("waypoint tree status: exists") 
        
                  
        closest_idx =  self.waypoint_tree.query([x,y], 1 )[1] # was ([x,y], 1 )[1] see below 
        #closest_idx =  10 #1 # a random test
    
        #from scipy website https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.query.html
        # usage KDTree.query(x, k=1, eps=0, p=2, distance_upper_bound=inf)[source]
        #check if closest is ahead or behind vehicle
        # I am going to copy this into tl_detector which is already slecting nearest waypoint but not nessecarily ahead waypoint might be causeing problems...John
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx -  1]
        
        #Equation of hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        val = np.dot(cl_vect - prev_vect , pos_vect - cl_vect )
        
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
            
        return closest_idx
        
    def publish_waypoints(self):#added self here after error message also closest)idx
        final_lane = self.generate_lane()
        # lane.header = self.base_waypoints.header
        # lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        #lane.waypoints = self.base_waypoints.waypoints[1: LOOKAHEAD_WPS] # a test
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header #moved from previous section to here
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        #to determine if need to decelerate, need to verify in simulator
        #testing what happens without if statement maybe stopline the problem
        # test interesting car stayed forzen swithed to manual drove it a little enaged auto and it stoped so the deceleration part works but not being trigger so problems must be around stopline_wp_idx or fartest_idx
        #lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        #another test 
        #self.stopline_wp_idx = Int32(self.stopline_wp_idx)#250 # this was close to the real value I want to see can I force it to stop
        #self.stopline_wp_idx = 292#similr to real value
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints 
            #experiment
            #lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)# I want to see will this cause it to stop
            
            #rospy.loginfo('In generate lane if statement:')
            rospy.loginfo('In generate lane if statement stopline is:{}'.format(self.stopline_wp_idx))
            
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            #rospy.loginfo('In generate lane else statement:')
            rospy.loginfo('In generate lane Else statement stopline is:{}'.format(self.stopline_wp_idx))

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        rospy.loginfo('In decelerate waypoints stopline:{}'.format(self.stopline_wp_idx))
        #john this is an attempt to save a before and after snapshot of waypoints to prove they are in fact getting modified
        #f_name = "Before.txt" #hmmm no too sure about txt....
        ##f = open(f_name,"w+")
        #f.write(waypoints)
        #f.close()
        
        
        temp = []
        for i, wp in enumerate(waypoints):

            p = Waypoint()
            p.pose = wp.pose

            # Distance includes a number of waypoints back so front of car stops at line
            # thi sline might be suspect so I am rewriting it....# stop_idx = max(self.stopline_wp_idx - closest_idx - STOP_LINE_MARGIN, 0)
            #john chnaged below added IF
            if( (self.stopline_wp_idx - closest_idx - STOP_LINE_MARGIN) <= 0 ):
                stop_idx = 0
            else: stop_idx = (self.stopline_wp_idx - closest_idx - STOP_LINE_MARGIN)
                
            #stop_idx = max(self.stopline_wp_idx - closest_idx - STOP_LINE_MARGIN, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist) + (i * CONSTANT_DECEL)
		   #or try the other way as below
		   #vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            #john below line showed X changing in logs so looks like its working
            #rospy.loginfo('In decelerate waypoints p =:{}'.format(p))
            temp.append(p)
        #not sure if the part below is useful, need to verify
        # self.decelerate_count += 1
        # if (self.decelerate_count % LOGGING_THROTTLE_FACTOR) == 0:
            # size = len(waypoints) - 1
            # vel_start = temp[0].twist.twist.linear.x
            # vel_end = temp[size].twist.twist.linear.x
            # rospy.logwarn("DECEL: vel[0]={:.2f}, vel[{}]={:.2f}".format(vel_start, size, vel_end))
            
            
        #john this is an attempt to save a before and after snapshot of waypoints to prove they are in fact getting modified
        #f_name = "After.txt" #hmmm no too sure about txt....
        #f = open(f_name,"w+")
        #f.write(temp)
        #f.close()
        return temp       

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg #.pose # I had only msg here!!
        #pass # i had not removed this might have been a problem

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        rospy.loginfo('In waypoints_cb :')
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [ [ waypoint.pose.pose.position.x , waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints ]
            self.waypoint_tree = KDTree(self.waypoints_2d)
      #  pass #left this in by mistake still gives tutle not integre error here

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # pass
        #From John next line added to get a value for self.stopline_wp_idx from /traffic_waypoint 
        self.stopline_wp_idx = msg.data #taking out data as experiment#putting it back in again much later re msg Int32 error
        rospy.loginfo('In traffic_cb self.stopline_wp_idx is : {}'.format(self.stopline_wp_idx) )
        #rospy.loginfo('Color pred {} with prob {}'.format(pred, prob))

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')