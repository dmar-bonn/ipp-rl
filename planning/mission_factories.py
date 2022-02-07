import logging
from typing import Dict

from constants import MISSION_TYPES, MissionParams, MissionType, UAV_PARAMS
from mapping.mappings import Mapping
from planning.baselines.conical_spiral_mission import ConicalSpiralMission
from planning.baselines.lawn_mower_mission import LawnMowerMission
from planning.baselines.random_continuous_mission import RandomContinuousMission
from planning.baselines.random_discrete_mission import RandomDiscreteMission
from planning.greedy_mission import GreedyMission
from planning.ipp_masha import IPPMashaMission
from planning.mcts_mission import MCTSMission
from planning.mcts_zero.mcts_zero_mission import MCTSZeroMission
from planning.missions import Mission

logger = logging.getLogger(__name__)


class MissionFactory:
    def __init__(self, params: Dict, mapping: Mapping, use_effective_mission_time: bool):
        self.params = params
        self.mapping = mapping
        self.use_effective_mission_time = use_effective_mission_time
        self.mission_params = self.get_mission_params()

    def get_mission_params(self) -> Dict:
        if self.mission_type not in MISSION_TYPES:
            logger.error(f"'{self.mission_type}' not in list of known missions: {MISSION_TYPES}")
            raise ValueError

        param_names = []
        if self.mission_type == MissionType.LAWNMOWER:
            param_names = MissionParams.STATIC_MISSION + MissionParams.LAWNMOWER
        elif self.mission_type == MissionType.CONICAL_SPIRAL:
            param_names = MissionParams.STATIC_MISSION + MissionParams.CONICAL_SPIRAL
        elif self.mission_type == MissionType.RANDOM_CONTINUOUS:
            param_names = MissionParams.STATIC_MISSION + MissionParams.RANDOM_CONTINUOUS
        elif self.mission_type == MissionType.RANDOM_DISCRETE:
            param_names = MissionParams.STATIC_MISSION + MissionParams.RANDOM_DISCRETE
        elif self.mission_type == MissionType.GREEDY:
            param_names = MissionParams.STATIC_MISSION + MissionParams.GREEDY
        elif self.mission_type == MissionType.MCTS:
            param_names = MissionParams.STATIC_MISSION + MissionParams.MCTS
        elif self.mission_type == MissionType.IPP_MASHA:
            param_names = MissionParams.STATIC_MISSION + MissionParams.IPP_MASHA
        elif self.mission_type == MissionType.MCTS_ZERO:
            param_names = MissionParams.STATIC_MISSION + MissionParams.MCTS_ZERO

        params = dict()
        for param in param_names:
            if isinstance(param, dict):
                for key in param.keys():
                    params[key] = {}
                    for sub_param in param[key]:
                        if sub_param not in self.params["mission"][key].keys():
                            logger.error(
                                f"Cannot find '{sub_param}' in '{key}' for mission '{self.mission_type}' in config file!"
                            )
                            raise ValueError

                        params[key][sub_param] = self.params["mission"][key][sub_param]
            else:
                if param not in self.params["mission"].keys():
                    logger.error(f"Cannot find '{param}' parameter for mission '{self.mission_type}' in config file!")
                    raise ValueError

                params[param] = self.params["mission"][param]

        params["mapping"] = self.mapping
        params["uav_specifications"] = self.get_uav_params()
        params["use_effective_mission_time"] = self.use_effective_mission_time

        return params

    def get_uav_params(self) -> Dict:
        params = dict()
        for param in UAV_PARAMS:
            if param not in self.uav_specifications.keys():
                logger.error(f"Cannot find '{param}' parameter for uav specification in config file!")
                raise ValueError

            params[param] = self.uav_specifications[param]

        return params

    @property
    def mission_type(self) -> str:
        if "mission" not in self.params.keys():
            logger.error("Cannot find mission specification in config file!")
            raise ValueError

        if "type" not in self.params["mission"].keys():
            logger.error("Cannot find mission type specification in config file!")
            raise ValueError

        return self.params["mission"]["type"]

    @property
    def uav_specifications(self) -> Dict:
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "uav" not in self.params["experiment"].keys():
            logger.error("Cannot find uav specification in config file!")
            raise ValueError

        return self.params["experiment"]["uav"]

    def create_mission(self) -> Mission:
        if self.mission_type not in MISSION_TYPES:
            logger.error(f"'{self.mission_type}' not in list of known mission types: {MISSION_TYPES}")
            raise ValueError

        if self.mission_type == MissionType.LAWNMOWER:
            return LawnMowerMission(**self.mission_params)
        elif self.mission_type == MissionType.CONICAL_SPIRAL:
            return ConicalSpiralMission(**self.mission_params)
        elif self.mission_type == MissionType.RANDOM_CONTINUOUS:
            return RandomContinuousMission(**self.mission_params)
        elif self.mission_type == MissionType.RANDOM_DISCRETE:
            return RandomDiscreteMission(**self.mission_params)
        elif self.mission_type == MissionType.GREEDY:
            return GreedyMission(**self.mission_params)
        elif self.mission_type == MissionType.MCTS:
            return MCTSMission(**self.mission_params)
        elif self.mission_type == MissionType.IPP_MASHA:
            return IPPMashaMission(**self.mission_params)
        elif self.mission_type == MissionType.MCTS_ZERO:
            return MCTSZeroMission(**self.mission_params)
