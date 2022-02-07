import logging
from typing import Dict

from constants import SENSOR_MODELS, SensorModelParams, SensorModelType
from sensors.models import SensorModel
from sensors.models.sensor_models import AltitudeSensorModel

logger = logging.getLogger(__name__)


class SensorModelFactory:
    def __init__(self, params: Dict):
        self.params = params
        self.model_params = self.get_model_params()

    def get_model_params(self) -> Dict:
        if self.sensor_model not in SENSOR_MODELS:
            logger.error(f"'{self.sensor_model}' not in list of known sensor models: {SENSOR_MODELS}")
            raise ValueError

        param_names = []
        if self.sensor_model == SensorModelType.ALTITUDE_DEPENDENT:
            param_names = SensorModelParams.ALTITUDE_DEPENDENT

        params = dict()
        for param in param_names:
            if param not in self.params["sensor"]["model"].keys():
                logger.error(f"Cannot find '{param}' parameter for sensor model '{self.sensor_model}' in config file!")
                raise ValueError

            params[param] = self.params["sensor"]["model"][param]

        return params

    @property
    def sensor_model(self) -> str:
        if "sensor" not in self.params.keys():
            logger.error("Cannot find sensor specification in config file!")
            raise ValueError

        if "model" not in self.params["sensor"].keys():
            logger.error("Cannot find sensor model specification in config file!")
            raise ValueError

        if "type" not in self.params["sensor"]["model"].keys():
            logger.error("Cannot find sensor model type specification in config file!")
            raise ValueError

        return self.params["sensor"]["model"]["type"]

    def create_sensor_model(self) -> SensorModel:
        if self.sensor_model not in SENSOR_MODELS:
            logger.error(f"'{self.sensor_model}' not in list of known sensor types: {SENSOR_MODELS}")
            raise ValueError

        if self.sensor_model == SensorModelType.ALTITUDE_DEPENDENT:
            return AltitudeSensorModel(**self.model_params)
