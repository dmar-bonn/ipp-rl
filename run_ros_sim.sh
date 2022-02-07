docker-compose down
docker-compose up -d
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' mapping_ipp_framework_mapping_ipp`
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' mapping_ipp_framework_rotors_simulation`
docker exec -it -d mapping_ipp_framework_rotors_simulation bash -c "source /mapping_ipp_framework/devel/setup.bash && roslaunch rotors_gazebo mav.launch mav_name:=firefly world_name:=basic"
docker exec -it -d mapping_ipp_framework_mav_local_planning bash -c "source /mapping_ipp_framework/devel/setup.bash && roslaunch mav_trajectory_generation_ros trajectory_sampler.launch"
docker exec -it mapping_ipp_framework_mav_ipp_planning bash -c "source /mapping_ipp_framework/devel/setup.bash && roslaunch ipp_planning mission.launch"
