#!/usr/bin/env python3

#Notes about HSV Values:
# H: Hue        ----- ranges from 0 to 180
# S: Saturation ----- ranges from 0 to 255
# V: Value      ----- ranges from 0 to 255

#import required modules

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from obj_detection.msg import floatList
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
import imutils 

bridge    = CvBridge()
rgb_img   = np.zeros((480,640,3),np.uint8)
depth_img = np.zeros((480,640))
fpoint=[]
a1=0
a2=0
a3=0
a4=0
######## UNCOMMENT TO USE TRACKBAR #######
def nothing(x):
    pass



#RGB Image Callback
def rgb_callback(rgb_msg):
    global rgb_img   
    rgb_img = bridge.imgmsg_to_cv2(rgb_msg, "bgr8")     


#Depth Image Callback 
def d_callback(msg):
    
    global depth_img
    depth_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
#Getting XY Points
def xy_points(frame):
    global a1,a2,a3,a4
    ######## UNCOMMENT TO USE TRACKBAR #######
    # global h_low,h_high,s_low,s_high,v_low,v_high    
    
    #initialise points array for 8 points
    xypoints = np.array([0,0,0,0,0,0,0,0], dtype = np.int64)

    # Gaussian Blur reduces noise, gives a better mask

    blurred  = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv      = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) #conversion to HSV
   
    #defining the Range of Blue color
    blue_lower = np.array([87,100,150],np.uint8)
    blue_upper = np.array([110,255,255],np.uint8)
    
    blue = cv2.inRange(hsv,blue_lower,blue_upper) 
    blue = cv2.erode(blue, None, iterations=2)
    blue = cv2.dilate(blue, None, iterations=2)
    
    # find Contours
    cnts_b   = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_b   = imutils.grab_contours(cnts_b)
    
    center_b = None

    # finds the biggest contour, computes centroid.
    if len(cnts_b) > 0:
        c_b = max(cnts_b, key = cv2.contourArea)
        M_b = cv2.moments(c_b)
        center_b = (int(M_b["m10"] / M_b["m00"]), int(M_b["m01"] / M_b["m00"]))
        cv2.circle(frame, center_b, 5, (0,0,0), -1)
        xypoints[2] = center_b[0]
        xypoints[3] = center_b[1]
    
    #Red
    low_r  = np.array([140,150,0],np.uint8)
    high_r = np.array([180,255,255],np.uint8)

    # low_r = np.array([170,220,220])
    low_r = np.array([0,45,142])
    high_r = np.array([180,255,255])

    min_red = np.array([0,45,142])
    max_red = np.array([180,255,255])

    # mask_r = cv2.inRange(hsv, min_red, max_red)
    # res_r = cv2.bitwise_and(frame,frame, mask= mask_r)
    # cv2.imshow('Red',res_r)

    red      = cv2.inRange(hsv,low_r,high_r)
    red      = cv2.erode(red, None, iterations=2)
    red      = cv2.dilate(red, None, iterations=2)
    
    cnts_r   = cv2.findContours(red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_r   = imutils.grab_contours(cnts_r)
    
    center_r = None
    if len(cnts_r) > 0:
    
        c_r = max(cnts_r, key=cv2.contourArea)
        M_r = cv2.moments(c_r)
    
        center_r = (int(M_r["m10"] / M_r["m00"]), int(M_r["m01"] / M_r["m00"]))
        cv2.circle(frame, center_r, 5, (0,0,0), -1)
    
        xypoints[4] = center_r[0]
        xypoints[5] = center_r[1]
    
    #Green
    low_g  = np.array([40,50,50])
    high_g = np.array([80,255,255])
    
    #GREEN_DETECTION
    #compute mask, erode and dilate it to remove noise
    green    = cv2.inRange(hsv, low_g, high_g)
    green    = cv2.erode(green, None, iterations=2)
    green    = cv2.dilate(green, None, iterations=2)
    cnts_g   = cv2.findContours(green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_g   = imutils.grab_contours(cnts_g)
    center_g = None
    
    if len(cnts_g) > 0:
        
        c_g = max(cnts_g, key=cv2.contourArea)
        
        # epsilon = 0.1*cv2.arcLength(c,True)
        # approx = cv2.approxPolyDP(c,epsilon,True)
        
        M_g = cv2.moments(c_g)
        center_g = (int(M_g["m10"] / M_g["m00"]), int(M_g["m01"] / M_g["m00"]))
        cv2.circle(frame, center_g, 5, (0,0,0), -1)
        
        xypoints[0] = center_g[0]
        xypoints[1] = center_g[1]
        
        cv2.circle(frame, (320,240), 5, (0,0,255), -1)      #ORIGIN
    
    
    #ORANGE
    
    
    #HSV VALUES FOR ORANGE
    low_o  = np.array([3,11,119])
    high_o = np.array([11,255,255])
    
    
    #ORANGE_DETECTION
    #compute mask, erode and dilate it to remove noise
    orange    = cv2.inRange(hsv, low_o, high_o)
    orange    = cv2.erode(orange, None, iterations=2)
    orange    = cv2.dilate(orange, None, iterations=2)
    cnts_o   = cv2.findContours(orange.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_o   = imutils.grab_contours(cnts_o)
    center_o = None
    if len(cnts_o) > 0:
        c_o = max(cnts_o, key=cv2.contourArea)

        #####TO DRAW THE CONTOUR####
        # epsilon = 0.1*cv2.arcLength(c,True)
        # approx = cv2.approxPolyDP(c,epsilon,True)
        

        M_o = cv2.moments(c_o)
        center_o = (int(M_o["m10"] / M_o["m00"]), int(M_o["m01"] / M_o["m00"]))
        cv2.circle(frame, center_o, 5, (0,0,0), -1)
        xypoints[6] = center_o[0]
        xypoints[7] = center_o[1]
        
    cv2.imshow("Color Tracking",frame)

    ###UNCOMMENT TO USE TRACKBAR####
    # cv2.imshow("orange_mask",orange)
    
    # When at the desired position, press "s" to record the feature points(in red)
    # After moving the robot to some other position, run the visual servoing node and watch the
    # black centroids merge with the red ones. 
    key=cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        a1=center_b
        a2=center_r
        a3=center_g
        a4=center_o
        a5=xypoints
    else:
        a5=xypoints
    
    if a1!=0:
        cv2.circle(frame, a1, 5, (0,0,255), -1)
        cv2.circle(frame, a2, 5, (0,0,255), -1)
        cv2.circle(frame, a3, 5, (0,0,255), -1)
        cv2.circle(frame, a4, 5, (0,0,255), -1)
        return a5
    else:
        return a5
        
def f_me(dimg, xy):

    fx = 347.9975
    cx = 320 
    fy =  347.9975
    cy= 240 

    fx = 640
    fy = 480

    # fx = 462.1379699707031
    # fy = fx

    u = xy[4]
    v = xy[5]

    # u = 345
    # v = 214
    print(u,v)
    z = dimg[int(v)][int(u)]

    x = (u-cx)*(z/fx)
    y = (v-cy)*(z/fy)

    print(x/1000,y/1000,z/1000)
    return x/1000,y/1000,z/1000

    
def f_points(dimg, xy):
    
    # cv2.imshow("Color Tracking",dimg)

    fl = 531.1   #525
    fpoint=[]
    
    #(369,256)
    # xy[4] = 369
    # xy[4+1] = 256

    # z = dimg[xy[5]][xy[4]]
    # fpoint.append(xy[4])
    # fpoint.append(xy[5])
    # fpoint.append(z)

    # fpoint[i] = -(fpoint[0]-320)/(fl/640)
    # fpoint[j] = -(fpoint[1]-240)/(fl/480)


    for i in range(0,len(xy),2):
        fpoint.append(xy[i])
        fpoint.append(xy[i+1])
        if xy[i+1]<320 and xy[i]<480:
            z = dimg[int(xy[i+1])][int(xy[i])]
            fpoint.append(z)
        else:
            z = dimg[320][240]
            fpoint.append(z)
      
    for i in range(0,len(fpoint),3):    
        fpoint[i] = -(fpoint[i]-320)/(fl/640)
    for j in range(1,len(fpoint),3):
        fpoint[j] = -(fpoint[j]-240)/(fl/480)
    print(fpoint)  
    return fpoint
 
def main():
    global rgb_img
    global depth_img
    
    rospy.init_node('d_points', anonymous=True)
    rospy.Subscriber('/camera/rgb/image_rect_color', Image, rgb_callback )
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, d_callback)
    # pub = rospy.Publisher('/point_features', floatList, queue_size = 1)
    pub = rospy.Publisher('/camera/object_track', PointStamped, queue_size = 1)
    # features = floatList()
       
    while not rospy.is_shutdown():
         
        xy = xy_points(rgb_img)
        # print(xy)     
        c = 0
        while c < 20:
            xy = xy + xy_points(rgb_img)
            c += 1
        xy = xy / c

        x,y,z=f_me(depth_img, xy)

        pointstamp = PointStamped()
        pointstamp.header.frame_id= 'camera_depth_optical_frame'
        pointstamp.header.stamp= rospy.Time(0)
        pointstamp.point.x= x
        pointstamp.point.y= y
        pointstamp.point.z= z

        pub.publish(pointstamp)

        
if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

