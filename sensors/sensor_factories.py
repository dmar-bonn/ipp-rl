import logging
from typing import Dict

from constants import SENSOR_TYPES, SensorParams, SensorType
from sensors import Sensor, cameras
from sensors.models import SensorModel
from mapping.grid_maps import GridMap

logger = logging.getLogger(__name__)


class SensorFactory:
    def __init__(self, params: Dict, sensor_model: SensorModel, grid_map: GridMap):
        """
        Factory creates sensor objects as specified in the config YAML-file.

        Args:
            params (Dict): configuration parameters as specified in YAML-file
            sensor_model (SensorModel): sensor model defining sensor measurement characteristics
            grid_map (GridMap): grid map representation of environment
        """
        self.params = params
        self.sensor_model = sensor_model
        self.grid_map = grid_map
        self.sensor_params = self.get_sensor_params()

    def get_sensor_params(self) -> Dict:
        if self.sensor_type not in SENSOR_TYPES:
            logger.error(f"'{self.sensor_type}' not in list of known sensor types: {SENSOR_TYPES}")
            raise ValueError

        param_names = []
        if self.sensor_type == SensorType.RGB_CAMERA:
            param_names = SensorParams.CAMERA + SensorParams.RGB_CAMERA

        params = dict()
        for param in param_names:
            if param not in self.params["sensor"].keys():
                logger.error(f"Cannot find '{param}' parameter for sensor type '{self.sensor_type}' in config file!")
                raise ValueError

            params[param] = self.params["sensor"][param]

        params["sensor_model"] = self.sensor_model
        params["grid_map"] = self.grid_map

        return params

    @property
    def sensor_type(self) -> str:
        if "sensor" not in self.params.keys():
            logger.error("Cannot find sensor specification in config file!")
            raise ValueError

        if "type" not in self.params["sensor"].keys():
            logger.error("Cannot find sensor type specification in config file!")
            raise ValueError

        return self.params["sensor"]["type"]

    def create_sensor(self) -> Sensor:
        if self.sensor_type not in SENSOR_TYPES:
            logger.error(f"'{self.sensor_type}' not in list of known sensor types: {SENSOR_TYPES}")
            raise ValueError

        if self.sensor_type == SensorType.RGB_CAMERA:
            return cameras.RGBCamera(**self.sensor_params)
