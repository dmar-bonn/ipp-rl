source /opt/ros/noetic/setup.bash

# prepare ssh key permissions
cp -R /tmp/.ssh /root/.ssh
chmod 700 /root/.ssh
chmod 644 /root/.ssh/id_rsa.pub
chmod 600 /root/.ssh/id_rsa
chmod 600 /root/.ssh/config

# initialize catkin workspace
catkin init
catkin config --extend /opt/ros/noetic
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin config --merge-devel

# download dependencies
mkdir -p src && cd src/
git clone git@gitlab.igg.uni-bonn.de:popoviclab/mav_comm.git
git clone git@gitlab.igg.uni-bonn.de:popoviclab/mav_trajectory_generation.git
git clone git@github.com:ethz-asl/rotors_simulator.git
git clone git@github.com:ethz-asl/mav_control_rw.git
git clone git@github.com:catkin/catkin_simple.git
git clone git@github.com:ethz-asl/eigen_catkin.git
git clone git@github.com:ethz-asl/eigen_checks.git
git clone git@github.com:ethz-asl/glog_catkin.git
git clone git@github.com:ethz-asl/nlopt.git

# move own packages to src file
cp -R ../planning/ .

# install workspace
echo "source /mapping_ipp_framework/devel/setup.bash" >> ~/.bashrc
source /opt/ros/noetic/setup.bash
# FIXME: include mav_control_rw if ROS NMPC control package should be built. This takes around 20min
catkin build # ipp_planning mav_trajectory_generation rotors_simulator mav_control_rw catkin_simple eigen_catkin eigen_checks glog_catkin mav_comm nlopt
