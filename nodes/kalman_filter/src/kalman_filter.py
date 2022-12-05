#!/usr/bin/env python3
from __future__ import print_function

import sys, os
import rospy
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseStamped
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import math
from scipy.spatial.transform import Rotation as R


# Populate the config dictionary with any
# configuration parameters that you need
config = {
    'sub_topic_name': "/orb_slam2_stereo/pose", 
    'pub_topic_name': "/kalman_filter/pose",
    'node_name': "kf",
    'rate': 2, # Hz
}

idx = 0
measurement = None


#########################################################
# HELPER FUNCIONS
#########################################################

def quat2euler(q):
    """Convert quaternion to Euler angles
    Parameters
    ----------
    q : array-like (4)
        Quaternion in form (qx, qy, qz, qw)
    
    Returns
    -------
    array-like (3)
        x,y,z Euler angles in radians (extrinsic)
    """
    r = R.from_quat(q)
    return r.as_euler('XYZ')


def euler2quat(x, y, z):
    """Convert Euler angles to quaternion
    
    Parameters
    ----------
    x, y, z : float
        x,y,z Euler angles in radians (extrinsic)
    
    Returns
    -------
    array-like (4)
        Quaternion in form (qx, qy, qz, qw)
    """
    r = R.from_euler('XYZ', [x, y, z])
    return r.as_quat()


###############################################################
# KALMAN FILTER
###############################################################

class KalmanFilter(object):
    def __init__(self, dim_x, dim_y):
        # modify depending on the tracked state
        # You may add additional methods to this class
        # for building ROS messages

        self.dt = 0.2 # b/c 5 Hz loop
        
        self.dim_x = dim_x # state dims
        self.dim_y = dim_y # measurement dims

        self.x = np.zeros((dim_x, 1)) # state
        self.P = np.eye(dim_x) # covariance
        self.Q = 0.1 * np.eye(dim_x) # propagation noise
        
        d = np.array([np.cos(self.x[4])*np.cos(self.x[5]),
                      np.cos(self.x[4])*np.sin(self.x[5]),
                      -np.sin(self.x[4])])
        self.F = np.eye(dim_x) # state transition matrix
        self.F[:3,-1] = self.dt * np.squeeze(d)
        
        self.H = np.block([np.eye(dim_y), np.zeros((dim_y,1))]) # measurement matrix
        self.R = 10 * np.eye(dim_y) # measurement uncertainty
        
    def predict(self):
        # prediction step
        d = np.array([np.cos(self.x[4])*np.cos(self.x[5]),
                      np.cos(self.x[4])*np.sin(self.x[5]),
                      -np.sin(self.x[4])])
        self.F[:3,-1] = self.dt * np.squeeze(d) # dynamics matrix update
        self.x = np.matmul(self.F, self.x) # state predict from dynamics
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q # convariance predict

    def update(self, y):
        # update step
        # y is a vector containing all the measurements

        # Kalman gain
        temp1 = np.matmul(self.P, self.H.T)
        temp2 = np.linalg.inv(self.R + np.matmul(np.matmul(self.H, self.P),self.H.T))
        K = np.matmul(temp1, temp2)
        self.x = self.x + np.matmul(K, (y - np.matmul(self.H,self.x))) # state update from measurement
        self.P = np.matmul((np.eye(self.dim_x) - np.matmul(K,self.H)), self.P) # covariance update


# Initialize Kalman Filter
kf = KalmanFilter(dim_x=7, dim_y=6)  # set dim_x and dim_y values appropriately

####################################################################
# ROS SUBSCRIBER
####################################################################

def callback(data):
    global idx, config, measurement
    rospy.loginfo(rospy.get_caller_id()+"   "+str(idx))
    idx += 1
    if idx == 1: # skip first value because it's bogus
        return
    # Read position and orientation from Pose message,
    # add noise and pass to measurement global variable
    x = data.pose.position.x + np.random.normal(0, 0.1)
    y = data.pose.position.y + np.random.normal(0, 0.1)
    z = data.pose.position.z + np.random.normal(0, 0.1)
    q0 = data.pose.orientation.w
    q1 = data.pose.orientation.x
    q2 = data.pose.orientation.y
    q3 = data.pose.orientation.z
    eu_ang = quat2euler([q1, q2, q3, q0])
    measurement = np.zeros((6,1))
    measurement[0], measurement[1], measurement[2] = x, y, z
    measurement[3], measurement[4], measurement[5] = eu_ang[0], eu_ang[1], eu_ang[2]

def subscribe(config):
    global idx, measurement
    # creating a subscriber for input and publisher for output 
    rospy.init_node(config['node_name'], anonymous=True)
    rospy.Subscriber(config['sub_topic_name'], PoseStamped, callback)
    kf_pose_pub = rospy.Publisher(config['pub_topic_name'], PoseStamped, queue_size=10)

    rate = rospy.Rate(config['rate'])
    
    kf_x_vec = []
    kf_y_vec = []
    kf_z_vec = []
    orb_x_vec = []
    orb_y_vec = []
    orb_z_vec = []

    # Replace with publisher loop which calls predict/update 
    # in Kalman filter and publishes the estimated pose
    while not rospy.is_shutdown():
        if measurement is None:
            rospy.loginfo("Subscribed SLAM pose is currently None")
            rate.sleep()
            continue

        # append to global vars for plotting purposes
        kf_x_vec.append(kf.x[0])
        kf_y_vec.append(kf.x[1])
        kf_z_vec.append(kf.x[2])
        orb_x_vec.append(measurement[0])
        orb_y_vec.append(measurement[1])
        orb_z_vec.append(measurement[2])
        
        # step Kalman Filter
        kf.predict()
        kf.update(measurement)

        # convert state to a pose msg
        msg = PoseStamped()
        msg.pose.position.x = kf.x[0]
        msg.pose.position.y = kf.x[1]
        msg.pose.position.z = kf.x[2]
        
        q = euler2quat(kf.x[3,0], kf.x[4,0], kf.x[5,0])
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        # publish pose msg
        kf_pose_pub.publish(msg)
        
        rospy.loginfo("Processed slam pose and image ".format(idx))

        rate.sleep()
    return kf_x_vec, kf_y_vec, kf_z_vec, orb_x_vec, orb_y_vec, orb_z_vec

if __name__ == '__main__':
    try:
        kf_x_vec, kf_y_vec, kf_z_vec, orb_x_vec, orb_y_vec, orb_z_vec = subscribe(config)
    except rospy.ROSInterruptException:
        pass
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(kf_x_vec, kf_y_vec, kf_z_vec)#, c=kf_z_vec, cmap='Greens', marker='o')
    ax.scatter3D(orb_x_vec, orb_y_vec, orb_z_vec)#, marker='+')
    ax.legend(['kalman filter','orb slam'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('ORB SLAM and Kalman Filter, Positions')
    plt.savefig("kf_vs_slam_3D.png")
    plt.show()

    fig = plt.figure()
    plt.plot(kf_x_vec, kf_y_vec)#, c=kf_z_vec, cmap='Greens', marker='o')
    plt.plot(orb_x_vec, orb_y_vec)#, marker='+')
    plt.legend(['kalman filter','orb slam'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('ORB SLAM and Kalman Filter, Positions')
    plt.savefig("kf_vs_slam_2D.png")
    plt.show()
