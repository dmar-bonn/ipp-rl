import logging
from typing import Dict

logger = logging.getLogger(__name__)


class GridMap:
    def __init__(self, params: Dict):
        self.params = params
        self.mean = None
        self.cov_matrix = None

    @property
    def x_dim(self) -> int:
        """Returns map's x-dimension in number of cells"""
        if "environment" not in self.params.keys():
            logger.error(f"Cannot find environment specification in config file!")
            raise ValueError

        if "x_dim" not in self.params["environment"].keys():
            logger.error(f"Cannot find environment's x_dim specification in config file!")
            raise ValueError

        return self.params["environment"]["x_dim"]

    @property
    def y_dim(self) -> int:
        """Returns map's y-dimension in number of cells"""
        if "environment" not in self.params.keys():
            logger.error(f"Cannot find environment specification in config file!")
            raise ValueError

        if "y_dim" not in self.params["environment"].keys():
            logger.error(f"Cannot find environment's y_dim specification in config file!")
            raise ValueError

        return self.params["environment"]["y_dim"]

    @property
    def resolution(self) -> int:
        """Returns grid resolution in [m/cell]"""
        if "environment" not in self.params.keys():
            logger.error(f"Cannot find environment specification in config file!")
            raise ValueError

        if "resolution" not in self.params["environment"].keys():
            logger.error(f"Cannot find environment's resolution specification in config file!")
            raise ValueError

        return self.params["environment"]["resolution"]

    @property
    def num_grid_cells(self):
        return self.x_dim * self.y_dim
