#!/usr/bin/env python3
import sys

sys.path.insert(0, "/mapping_ipp_framework")

import rospy
from geometry_msgs.msg import Point
from mav_planning_msgs.msg import WaypointsTrajectory
from std_msgs.msg import Header

import constants
from config.params import load_params
from logger import setup_logger
from mapping.grid_maps import GridMap
from mapping.mappings import Mapping
from planning.mission_factories import MissionFactory
from sensors.models.sensor_model_factories import SensorModelFactory
from sensors.sensor_factories import SensorFactory
from simulations.simulation_factories import SimulationFactory


def execute_ipp_mission():
    logger = setup_logger()
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)

    logger.info("\n-------------------------------------- START MISSION --------------------------------------\n")

    waypoints_pub = rospy.Publisher("plan/waypoints", WaypointsTrajectory, queue_size=1, latch=True)
    rospy.init_node("ipp_mission")

    grid_map = GridMap(params)

    sensor_model_factory = SensorModelFactory(params)
    sensor_model = sensor_model_factory.create_sensor_model()

    sensor_factory = SensorFactory(params, sensor_model, grid_map)
    sensor = sensor_factory.create_sensor()

    sensor_simulation_factory = SimulationFactory(params, sensor)
    sensor_simulation = sensor_simulation_factory.create_sensor_simulation()
    sensor.set_sensor_simulation(sensor_simulation)
    sensor_simulation.visualize_ground_truth_map()

    mapping = Mapping(grid_map, sensor)

    mission_factory = MissionFactory(params, mapping)
    mission = mission_factory.create_mission()

    mission.waypoints = mission.create_waypoints()
    mission.visualize_path()

    waypoints_msg = WaypointsTrajectory()
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "world"
    waypoints_msg.header = header
    waypoints_msg.max_v = mission.uav_specifications["max_v"]
    waypoints_msg.max_a = mission.uav_specifications["max_a"]
    waypoints_msg.sampling_time = mission.uav_specifications["sampling_time"]

    for i in range(mission.waypoints.shape[0]):
        point = Point()
        point.x = mission.waypoints[i][0]
        point.y = mission.waypoints[i][1]
        point.z = mission.waypoints[i][2]
        waypoints_msg.waypoints.append(point)

    waypoints_pub.publish(waypoints_msg)


if __name__ == "__main__":
    execute_ipp_mission()
