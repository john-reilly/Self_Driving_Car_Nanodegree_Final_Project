#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        #I added min_speed here extra to video but hardcoded else where so I put it here as a variable to be reset if needed
        min_speed =  0.1 # 10 #0.1

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # TODO: Create `Controller` object
        # self.controller = Controller(<Arguments you wish to provide>)
        # i missed this got error evenutally about controller
        # all atribute from above
        '''
        self.controller = Controller( vehicle_mass = vehicle_mass , # i thkn i am not using kwargs correctly https://pythontips.com/2013/08/04/args-and-kwargs-in-python-explained/
                                        fuel_capacity = fuel_capacity ,
                                        brake_deadband = brake_deadband ,
                                        decel_limit = decel_limit ,
                                        accel_limit =  accel_limit ,
                                        wheel_radius = wheel_radius ,
                                        wheel_base = wheel_base ,
                                        steer_ratio = steer_ratio ,
                                        max_lat_accel =  max_lat_accel ,
                                        max_steer_angle = max_steer_angle,
                                        min_speed = min_speed)# I added min_speed here was hardcoced in video as 0.1
       
        self.controller = Controller( vehicle_mass = 'vehicle_mass' ,
                                        fuel_capacity = 'fuel_capacity' ,
                                        brake_deadband = 'brake_deadband' ,
                                        decel_limit = 'decel_limit' ,
                                        accel_limit =  'accel_limit' ,
                                        wheel_radius = 'wheel_radius' ,
                                        wheel_base = 'wheel_base' ,
                                        steer_ratio = 'steer_ratio' ,
                                        max_lat_accel =  'max_lat_accel' ,
                                        max_steer_angle = 'max_steer_angle',
                                        min_speed = 'min_speed')
        '''
        #another attempt around **kwargs##belwo seems to work
        self.controller = Controller( vehicle_mass  ,
                                        fuel_capacity ,
                                        brake_deadband ,
                                        decel_limit  ,
                                        accel_limit  ,
                                        wheel_radius ,
                                        wheel_base  ,
                                        steer_ratio ,
                                        max_lat_accel ,
                                        max_steer_angle,
                                        min_speed )
        

        # TODO: Subscribe to all the topics you need to
        #from Q+A
        rospy.Subscriber('/vehicle/dbw_enabled',Bool,self.dbw_enabled_cb)#turning off for a test
        rospy.Subscriber('/twist_cmd',TwistStamped,self.twist_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        
        self.current_vel = None
        self.current_ang_vel = None
        self.dbw_enabled =  False # False # None # should this be false?
        self.linear_vel = None
        self.angular_vel =  None
        self.throttle = self.steering = self.brake = 0
        

        self.loop()

    def loop(self):
        rate = rospy.Rate(50) # maybe small difference at 10# 20 made no apparent difference #original value 50 I am trying to fix the passing the green dots problem by adjusting this rate# 50Hz
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            if not None in (self.current_vel ,self.linear_vel, self.angular_vel):
                self.throttle, self.brake, self.steering = self.controller.control(self.current_vel, 
                                                                               self.dbw_enabled,
                                                                               self.linear_vel,
                                                                               self.angular_vel)
                
                #self.throttle = 1# testing to see it is executing in here and it is
                #self.brake =  0
                #self.steering = 0.5#this much worked..test
          
            #throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)
            #self.dbw_enabled =  False #True# a test It does seem to go in here and execute
            if self.dbw_enabled : # changed== True : #was this in video <dbw is enabled>:
              self.publish(self.throttle, self.brake, self.steering)
              #self.publish(1,0,0)#this much worked..test
            rate.sleep()

    def dbw_enabled_cb(self,msg):#from Q+A
        self.dbw_enabled = msg
    
    def twist_cb(self,msg):# from Q+A
        self.linear_vel = msg.twist.linear.x
        self.angular_vel = msg.twist.angular.z # had a typo here
        
    
    def velocity_cb(self,msg):#from Q+A
        self.current_vel =  msg.twist.linear.x
        
        
    
    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
