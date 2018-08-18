import rospy
from std_msgs.msg import String

def callback(data):	
    rospy.loginfo(data)

def listener():

    rospy.init_node('listener', anonymous=True)
    inception_output', String, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
