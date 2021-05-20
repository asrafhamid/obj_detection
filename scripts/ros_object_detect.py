#!/usr/bin/env python3


import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from obj_detection.msg import floatList
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
import imutils
import math

class ColorObj:
    def __init__(self,name,low_r=np.array([0,0,0],np.uint8),high_r=np.array([0,0,0],np.uint8),low_c=None,high_c=None):
        self.low_r = low_r
        self.high_r = high_r

        self.low_c = low_c
        self.high_c = high_c

        self.xyz = np.array([0.0,0.0,0.0])
        self.name = name


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

        res_r = cv2.bitwise_and(frame,frame, mask= self.color)
        # cv2.imshow('Red',res_r)
        
        self.color = cv2.erode(self.color, None, iterations=2)
        self.color = cv2.dilate(self.color, None, iterations=2)

        return res_r


class ObjectDetector:
    def __init__(self):

        self.bridge    = CvBridge()
        self.rgb_img   = np.zeros((480,640,3),np.uint8)
        self.depth_img = np.zeros((480,640))
        self.xypoints = np.array([0,0,0,0,0,0,0,0], dtype = np.int64)
        self.clr_list = []

        self.sub_rgb = rospy.Subscriber('/camera/rgb/image_rect_color', Image, self.rgb_callback)
        # self.sub_rgb = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        self.sub_depth = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.d_callback)

        self.pub = rospy.Publisher('/camera/object_track', PoseStamped, queue_size = 1)

    
    def rgb_callback(self,rgb_msg):
        self.rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")

        if self.clr_list:
            for clr in self.clr_list:
                # self.detect_color(clr)
                self.detect_object(clr)
            
            self.pub_xyz(self.clr_list[0])


    def d_callback(self,msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    
    def add_color_obj(self,c_obj):
        self.clr_list.append(c_obj)


    def detect_color(self,color_obj):
        frame = self.rgb_img

        res_r = color_obj.process_color(frame)
        color = color_obj.color
        
        cnts_c   = cv2.findContours(color.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_c   = imutils.grab_contours(cnts_c)
        
        center_c = None
        if len(cnts_c) > 0:
        
            c_c = max(cnts_c, key=cv2.contourArea)
            M_c = cv2.moments(c_c)
        
            center_c = (int(M_c["m10"] / M_c["m00"]), int(M_c["m01"] / M_c["m00"]))
            cv2.circle(frame, center_c, 5, (0,0,0), -1)

            self.xypoints[0] = center_c[0]
            self.xypoints[1] = center_c[1]

            self.calc_obj_pos(color_obj)

        # cv2.imshow("Color Tracking",frame)
        # cv2.waitKey(1)

    def detect_object(self,color_obj):
        frame = self.rgb_img

        img = color_obj.process_color(frame)
        color = color_obj.color

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for i, c in enumerate(contours):
 
            area = cv2.contourArea(c)
            # print(area)
            
            if area < 1000 or 50000 < area:
                continue
            
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
            
            shape = "unidentified"
            ar = width / float(height)
            
            if width < height:
                angle = 1.5708 - angle
            else:
                angle = -angle

            if ar >= 0.85 and ar <= 1.15:
                shape = "square"
                angle = int(0)
            else:
                shape = "rectangle"
                    
            label = " Angle: " + str(angle) + " deg,"+str(center)
            # label = " Angle: " + str(angle) + " deg,"+str(shape)
            textbox = cv2.rectangle(img, (center[0]-35, center[1]-25), 
                (center[0] + 295, center[1] + 10), (255,255,255), -1)
            cv2.putText(img, label, (center[0]-50, center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
            cv2.drawContours(img,[box],0,(0,0,255),2)

            self.xypoints[0] = center[0]
            self.xypoints[1] = center[1]
            self.angle= angle

            self.calc_obj_pos(color_obj)
            
        cv2.imshow('Output Image', img)
        cv2.waitKey(1)

    def calc_obj_pos(self,color_obj):
        cx,cy = 320, 240 
        fx,fy = 640, 480

        u = self.xypoints[0]
        v = self.xypoints[1]

        print(u,v)
        z = self.depth_img[int(v)][int(u)]

        # converted x,y 
        x = (u-cx)*(z/fx)
        y = (v-cy)*(z/fy)

        color_obj.xyz[0] = x
        color_obj.xyz[1] = y
        color_obj.xyz[2] = z

        color_obj.xyz = color_obj.xyz/1000

        print(x/1000,y/1000,z/1000)


    def pub_xyz(self,clr):

        posestamp = PoseStamped()
        posestamp.header.frame_id= 'camera_depth_optical_frame'
        posestamp.header.stamp= rospy.Time(0)
        posestamp.pose.position.x = clr.xyz[0]
        posestamp.pose.position.y = clr.xyz[1]
        posestamp.pose.position.z = clr.xyz[2]
        posestamp.pose.orientation.w = self.angle

        self.pub.publish(posestamp)


def main():
    pass

        
if __name__=='__main__':
    try:
        rospy.init_node('object_detector', anonymous=True)

        #Notes about HSV Values:
        # H: Hue        ----- ranges from 0 to 180
        # S: Saturation ----- ranges from 0 to 255
        # V: Value      ----- ranges from 0 to 255
        # low_r  = np.array([0,45,142],np.uint8)
        # high_r = np.array([180,255,255],np.uint8)

        low_r  = np.array([0,150,70],np.uint8)
        high_r = np.array([180,255,255],np.uint8)
        
        low_c  = np.array([155,50,50],np.uint8)
        high_c = np.array([180,255,255],np.uint8)

        red = ColorObj("red",low_r,high_r,low_c,high_c)

        obj_det = ObjectDetector()
        obj_det.add_color_obj(red)

        rospy.spin()

        # main()
    except rospy.ROSInterruptException:
        pass

