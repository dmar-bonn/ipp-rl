import logging
from typing import Dict

from constants import SENSOR_SIMULATIONS, SensorSimulationParams, SensorSimulationType
from sensors import Sensor
from simulations import Simulation
from simulations.simulations import GaussianRandomField, HotspotRandomField, SplitRandomField, TemperatureDataField

logger = logging.getLogger(__name__)


class SimulationFactory:
    def __init__(self, params: Dict, sensor: Sensor):
        self.params = params
        self.sensor = sensor
        self.simulation_params = self.get_simulation_params()

    def get_simulation_params(self) -> Dict:
        if self.sensor_simulation not in SENSOR_SIMULATIONS:
            logger.error(f"'{self.sensor_simulation}' not in list of known sensor simulations: {SENSOR_SIMULATIONS}")
            raise ValueError

        param_names = []
        if self.sensor_simulation == SensorSimulationType.GAUSSIAN_RANDOM_FIELD:
            param_names = SensorSimulationParams.GAUSSIAN_RANDOM_FIELD
        elif self.sensor_simulation == SensorSimulationType.HOTSPOT_RANDOM_FIELD:
            param_names = SensorSimulationParams.HOTSPOT_RANDOM_FIELD
        elif self.sensor_simulation == SensorSimulationType.SPLIT_RANDOM_FIELD:
            param_names = SensorSimulationParams.SPLIT_RANDOM_FIELD
        elif self.sensor_simulation == SensorSimulationType.TEMPERATURE_DATA_FIELD:
            param_names = SensorSimulationParams.TEMPERATURE_DATA_FIELD

        params = dict()
        for param in param_names:
            if param not in self.params["sensor"]["simulation"].keys():
                logger.error(
                    f"Cannot find '{param}' parameter for sensor simulation '{self.sensor_simulation}' in config file!"
                )
                raise ValueError

            params[param] = self.params["sensor"]["simulation"][param]

        params["sensor"] = self.sensor

        return params

    @property
    def sensor_simulation(self) -> str:
        if "sensor" not in self.params.keys():
            logger.error("Cannot find sensor specification in config file!")
            raise ValueError

        if "simulation" not in self.params["sensor"].keys():
            logger.error("Cannot find sensor simulation specification in config file!")
            raise ValueError

        if "type" not in self.params["sensor"]["simulation"].keys():
            logger.error("Cannot find sensor simulation type specification in config file!")
            raise ValueError

        return self.params["sensor"]["simulation"]["type"]

    def create_sensor_simulation(self) -> Simulation:
        if self.sensor_simulation not in SENSOR_SIMULATIONS:
            logger.error(f"'{self.sensor_simulation}' not in list of known simulation types: {SENSOR_SIMULATIONS}")
            raise ValueError

        if self.sensor_simulation == SensorSimulationType.GAUSSIAN_RANDOM_FIELD:
            return GaussianRandomField(**self.simulation_params)
        elif self.sensor_simulation == SensorSimulationType.HOTSPOT_RANDOM_FIELD:
            return HotspotRandomField(**self.simulation_params)
        elif self.sensor_simulation == SensorSimulationType.SPLIT_RANDOM_FIELD:
            return SplitRandomField(**self.simulation_params)
        elif self.sensor_simulation == SensorSimulationType.TEMPERATURE_DATA_FIELD:
            return TemperatureDataField(**self.simulation_params)
