import cv2
import numpy
import rospy
import cv_bridge
from sensor_msgs.msg import Image

cam= cv2.VideoCapture(0)
bridge = cv_bridge.CvBridge()

pub = rospy.Publisher('inception_input', Image, queue_size=10)
rospy.init_node('camera', anonymous=True)
rate = rospy.Rate(6) #runs at 10 hz to avoid flooding the NN

mean = 128.0
std = 1.0/128.0

while not rospy.is_shutdown():
    ret, cvimg=cam.read()

    msg = bridge.cv2_to_imgmsg(cvimg)
    pub.publish(msg)
rate.sleep()