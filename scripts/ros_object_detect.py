#!/usr/bin/env python3


import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
import message_filters
from message_filters import Subscriber
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseArray
from obj_detection.srv import GetObject
import math
import tf
from geometry_msgs.msg import PointStamped, PoseStamped, PoseArray, Pose
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import numpy.ma as ma


class ColorObj:
    def __init__(self,name,low_r,high_r,low_c=None,high_c=None):
        self.low_r = low_r
        self.high_r = high_r

        self.low_c = low_c
        self.high_c = high_c

        # self.xyz = np.array([0.0,0.0,0.0])
        self.name = name
        self.color = None
        self.poses = PoseArray()


    def set_range(self,low_r,high_r):
        self.low_r = low_r
        self.high_r = high_r

    
    def process_color(self,frame):

        # Gaussian Blur reduces noise, gives a better mask
        blurred  = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv      = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #conversion to HSV

        self.color = cv2.inRange(hsv, self.low_r, self.high_r)

        if self.low_c is not None:
            color = cv2.inRange(hsv, self.low_c, self.high_c)
            self.color = self.color + color
            # self.color = cv2.erode(self.color, None, iterations=2)
            # self.color = cv2.dilate(self.color, None, iterations=2)

        res_r = cv2.bitwise_and(frame,frame, mask= self.color)
        # cv2.imshow('Red',res_r)
        
        return res_r


class ObjectDetector:
    def __init__(self):

        self.bridge    = CvBridge()
        self.rgb_img   = np.zeros((720,1280,3),np.uint8)
        self.depth_img = np.zeros((720,1280))
        self.xypoints = np.array([0,0,0,0,0,0,0,0], dtype = np.int64)
        self.clr_list = []
        self.clr_poses= [] 

        # self.sub_rgb = Subscriber('/camera/rgb/image_rect_color', Image)
        self.sub_rgb = Subscriber('/camera/color/image_raw', Image)
        # change: use aligned depth to color to get correct depth 
        self.sub_depth = Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

        self.sub = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.5)
        self.sub.registerCallback(self.callback)

        self.pub_detect = rospy.Publisher('/camera/object_detect', Image, queue_size = 1)
        self.pub_tf = rospy.Publisher('/object_tf', PoseArray, queue_size = 1)

        self.serv = rospy.Service("/get_obj_clr", GetObject, self.get_obj)


    def get_tf_obj(self):
        # print("ALL")
        poses = PoseArray()
        
        for clr in self.clr_list:
            if clr.poses:
                poses.header.frame_id = 'camera_depth_frame'
                poses.header.stamp = rospy.Time(0)
                poses.poses += clr.poses.poses

        self.pub_tf.publish(poses)

    def get_obj(self,req):
        if req.color == 'all':
            print("ALL")
            poses = PoseArray()
            
            for clr in self.clr_list:
                if clr.poses.poses:
                    poses.header.frame_id = 'camera_depth_frame'
                    poses.header.stamp = rospy.Time(0)
                    poses.poses += clr.poses.poses

            return(True, poses)

        for clr in self.clr_list:
            if clr.name == req.color:
               if clr.poses.poses:
                   clr.poses.header.frame_id = 'camera_depth_frame'
                   clr.poses.header.stamp = rospy.Time(0)
                   return (True,clr.poses)

        return(False, PoseArray())


    def callback(self,img,depth):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
            self.rgb_img = self.bridge.imgmsg_to_cv2(img, "bgr8")

            if self.clr_list:
                for clr in self.clr_list:
                    # create a rectangle over object, sets obj position and orientation
                    self.detect_object(clr)
                
        except CvBridgeError as e:
                print(e)
    

    def add_color_obj(self,c_obj):
        self.clr_list.append(c_obj)


    def detect_object(self,color_obj):
        frame = self.rgb_img

        img = color_obj.process_color(frame)
        # cv2.imshow(color_obj.name, img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if cv2.__version__ == "3.2.0":
            _,contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:
            contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # clear poses for this color
        color_obj.poses.poses.clear()

        for i, c in enumerate(contours):
 
            area = cv2.contourArea(c)
            center = [0.00,0.00]
            

            # calc only if area is big enuff
            if area > 600 and area < 50000:
                
                # print("area: "+str(area))
                # cv.minAreaRect returns:
                # (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(c)
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Retrieve the key parameters of the rotated bounding box
                center = (int(rect[0][0]),int(rect[0][1])) 
                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = int(rect[2])
                angle = math.radians(angle)
                
                shape = "na"
                ar = width / float(height)


                if width < height:
                    angle = angle + 1.5708
                else:
                    angle = angle

                # check if circle
                approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
                # print("contour len: {}".format(len(approx)))
                k = cv2.isContourConvex(approx)
                if k:
                    # print("circle")
                    angle = 1.5708
                    shape = "circle"
                else:
                    shape = "rect"

                label = "Angle: {:.2f}, Shape: {}".format(angle,shape)
                # textbox = cv2.rectangle(frame, (center[0]-35, center[1]-25),
                #     (center[0] + 295, center[1] + 10), (255,255,255), -1)
                cv2.putText(frame, label, (center[0]-50, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)

                red = [0,0,255]

                cv2.circle(frame, (center[0],center[1]), 1, red, -1)
                cv2.drawContours(frame,[box],0,(0,0,255),2)

                self.calc_obj_pos(color_obj,center[0],center[1],angle)
                self.get_tf_obj()
            
        try:
            self.pub_detect.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
            print(e)

        # cv2.imshow(color_obj.name, img)
        cv2.waitKey(1)

    def calc_obj_pos(self,color_obj,u,v,angle):
        cx,cy = 333.1079, 243.9019
        fx,fy = 616.889, 617.045

        print(u,v)
        z = self.depth_img[int(v),int(u)]

        v = int(v)
        u = int(u)
        # c = 60
        
        mx = ma.masked_array(self.depth_img, mask=self.depth_img==0)
        mx.min(1)
        z = mx[v,u]
        # z_arr = mx[v-c:v+c,u-c:u+c]
        print("v: {}, u: {}".format(int(v) ,int(u)))
        # print("max:{}, min:{}, real: {}".format(np.max(z_arr),np.min(z_arr),z))

        # try:
        #     # print("---",z)
        #     print("max:{}, min:{}, real: {}".format(np.max(z_arr),np.min(z_arr),z))
        # except ValueError:  #raised if `z` is empty. Why z can be empty?
        #     pass


        # converted x,y 
        x = (u-cx)*(z/fx)
        y = (v-cy)*(z/fy)

        pose = Pose()
        pose.position.x = x/1000
        pose.position.y = y/1000
        pose.position.z = z/1000
        # pose.orientation.w = angle

        rot = quaternion_from_euler(0,0,angle)

        pose.orientation.x = rot[0]
        pose.orientation.y = rot[1]
        pose.orientation.z = rot[2]
        pose.orientation.w = rot[3]

        color_obj.poses.poses.append(pose)


        # print(x/1000,y/1000,z/1000)
        print("x: {:.4f}, y: {:.4f}, z: {:.4f}".format(x/1000,y/1000,z/1000))

        
if __name__=='__main__':
    try:
        rospy.init_node('object_detector', anonymous=True)
        # listener = tf.TransformListener()
        # listener.waitForTransform("/base_link", "/camera_link", rospy.Time(0),rospy.Duration(4.0))


        #Notes about HSV Values:
        # H: Hue        ----- ranges from 0 to 180
        # S: Saturation ----- ranges from 0 to 255
        # V: Value      ----- ranges from 0 to 255
        # low_r  = np.array([0,45,142],np.uint8)
        # high_r = np.array([180,255,255],np.uint8)

        low_r  = np.array([0,50,50],np.uint8)
        high_r = np.array([10,255,255],np.uint8)

        low_b  = np.array([100,150,0],np.uint8)
        high_b = np.array([140,255,255],np.uint8)
        
        low_c  = np.array([170,50,50],np.uint8)
        high_c = np.array([180,255,255],np.uint8)

        red = ColorObj("red",low_r,high_r,low_c,high_c)
        blue = ColorObj("blue",low_b,high_b)

        obj_det = ObjectDetector()

        obj_det.add_color_obj(red)
        obj_det.add_color_obj(blue)

        rospy.spin()

    except rospy.ROSInterruptException:
        pass

