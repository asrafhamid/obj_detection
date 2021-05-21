# obj_detection


## Overview

A ros_wrapper of opencv realtime object detection to detect single colored object with realsense depth camera
- Reads in depth and color image from [realsense](https://github.com/IntelRealSense/realsense-ros)

**Keywords:** simple object detection, object tracking, realsense camera


**Author: Mohd Asraf <br />

The obj_detection package has been tested under [ROS] Melodic on Ubuntu 18.04.
This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.


![Example image](doc/example.jpg)


## Installation

### Building from Source

#### Dependencies

- [Robot Operating System (ROS)](http://wiki.ros.org) (middleware for robotics),
- [CV bridge](http://wiki.ros.org/cv_bridge) (Packages for interfacing ROS with OpenCV)
 
Install following python depencies

	sudo apt-get install python3-opencv
	sudo pip3 install imutils

- opencv: 3.2.0
- imutils: 0.5.4

#### Building

I suggest creating a new catkin workspace for working with python3 packages

	source /opt/ros/melodic/setup.bash
	mkdir -p ~/vision_ws/src
	cd vision_ws/
	catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
  
To build from source, clone the dependencies / obj_detection repo into your catkin workspace and catkin build

	cd ~/vision_ws/src
	git clone -b melodic https://github.com/ros-perception/vision_opencv.git
	git clone -b code_review https://github.com/aychaplin/obj_detection.git
	cd ../
	catkin build


## Usage

  
Run object_detector node with:

	rosrun obj_detection ros_object_detect.py


## Nodes

### object_detector

Subscribes to depth camera topics, filters by HSV range and publishes object position.


#### Subscribed Topics

* **`/camera/rgb/image_rect_color`** ([sensor_msgs/Image])

	color image.
  
* **`/camera/depth/image_rect_raw`** ([sensor_msgs/Image])

	depth image.

#### Published Topics

* **`/camera/object_track`** ([geometry_msgs/PoseStamped])

	object pose.
  
* **`/camera/object_detect`** ([sensor_msgs/Image])

	filtered object image.

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/aychaplin/obj_detection/issues).

[ROS]: http://www.ros.org
[geometry_msgs/PoseStamped]: http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/PoseStamped.html
