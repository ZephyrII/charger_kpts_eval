## Installation:

Start with docker image tensorflow/tensorflow:1.15.4-gpu-py3:

 - `docker pull tensorflow/tensorflow:1.15.4-gpu-py3`
 -  `docker run --runtime=nvidia -it --env="DISPLAY"   
    --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="/path/to/shared/dir/on/host:/path/to/shared/dir/inside/container"
    --name="NameOfYourContainer" --net=host tensorflow/tensorflow:1.15.4-gpu-py3 bash`

Install ROS-melodic and other dependencies([http://wiki.ros.org/melodic/Installation/Ubuntu](http://wiki.ros.org/melodic/Installation/Ubuntu)):

- `sh -c 'echo "deb [http://packages.ros.org/ros/ubuntu](http://packages.ros.org/ros/ubuntu) $(lsb_release -sc) main" /etc/apt/sources.list.d/ros-latest.list'`
 - `apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654`
- `apt update`
- `apt install python-catkin-tools python3-opencv python3-rospkg git`
- `apt install ros-melodic-desktop`
- `apt install ros-melodic-gps-common`
- `echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc`
- `source ~/.bashrc`

Create workspace using catkin build:

- `mkdir -p ~/catkin_ws/src`
- `cd ~/catkin_ws/`
- `catkin build`
- `cd src`

Clone repositories. Get URLs from your bitbucket account:

- `git clone https://nowaktomasz@bitbucket.org/put_adas_ub/keras_charger_detector.git`
- `git clone https://nowaktomasz@bitbucket.org/put_adas_ub/custom_solvepnp.git`
- `catkin build custom_solvepnp_msg deep_pose_estimator`

Install python packages:

- `pip3 install keras==2.1.0`
- `pip3 install scikit-image scikit-learn`
- `source ~/catkin_ws/devel/setup.bash`

Run node:

- `roslaunch deep_pose_estimator detector.launch`