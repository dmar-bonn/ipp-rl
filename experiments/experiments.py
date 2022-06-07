import copy
import logging
import os
import pickle
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from constants import (
    BASELINE_MISSION_TYPES,
    EXPERIMENTS_FOLDER,
    EvaluationMeasure,
    PLOT_LABEL_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    PLOT_LINE_WIDTH,
    PLOT_TICKS_SIZE,
    MissionType,
)
from mapping.grid_maps import GridMap
from mapping.mappings import Mapping
from planning.mission_factories import MissionFactory
from sensors.models.sensor_model_factories import SensorModelFactory
from sensors.sensor_factories import SensorFactory
from simulations.simulation_factories import SimulationFactory

sns.set()
logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, params: Dict):
        self.params = params

        self.constraints_params = self.get_constraints_params()
        self.scenario_params = self.get_scenario_params()
        self.missions_params = self.get_missions_params()
        self.evaluation_metrics = self.get_evaluation_metrics()
        self.repetitions = self.get_repetitions()
        self.use_effective_mission_time = self.get_effective_mission_time_mode()
        self.title = self.get_title()

        self.missions = {}
        self.mappings = {}
        self.setup()

        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.save_folder = os.path.join(EXPERIMENTS_FOLDER, f"{self.title}_{timestamp}")

    def get_color_palette(self):
        color_palette = {}
        for mission_type in self.missions.keys():
            mission_color = self.missions[mission_type]["params"]["mission"]["color"]
            mission_label = self.missions[mission_type]["instances"][0].mission_label
            color_palette[mission_label] = mission_color

        return color_palette

    def get_evaluation_metrics(self):
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "evaluation" not in self.params["experiment"].keys():
            logger.error("Cannot find experiment evaluation specification in config file!")
            raise ValueError

        if "metrics" not in self.params["experiment"]["evaluation"].keys():
            logger.error("Cannot find experiment evaluation metrics specification in config file!")
            raise ValueError

        return self.params["experiment"]["evaluation"]["metrics"]

    def get_repetitions(self):
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "evaluation" not in self.params["experiment"].keys():
            logger.error("Cannot find experiment evaluation specification in config file!")
            raise ValueError

        if "repetitions" not in self.params["experiment"]["evaluation"].keys():
            logger.error("Cannot find experiment repetitions specification in config file!")
            raise ValueError

        return self.params["experiment"]["evaluation"]["repetitions"]

    def get_effective_mission_time_mode(self):
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "evaluation" not in self.params["experiment"].keys():
            logger.error("Cannot find experiment evaluation specification in config file!")
            raise ValueError

        if "use_effective_mission_time" not in self.params["experiment"]["evaluation"].keys():
            logger.error("Cannot find experiment effective_mission_time mode specification in config file!")
            raise ValueError

        return self.params["experiment"]["evaluation"]["use_effective_mission_time"]

    def get_title(self):
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "title" not in self.params["experiment"].keys():
            logger.error("Cannot find experiment title specification in config file!")
            raise ValueError

        return self.params["experiment"]["title"]

    def get_constraints_params(self):
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "constraints" not in self.params["experiment"].keys():
            logger.error("Cannot find experiment constraints specification in config file!")
            raise ValueError

        return self.params["experiment"]["constraints"]

    def get_scenario_params(self):
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "scenario" not in self.params["experiment"].keys():
            logger.error("Cannot find experiment scenario specification in config file!")
            raise ValueError

        return self.params["experiment"]["scenario"]

    def get_missions_params(self):
        if "experiment" not in self.params.keys():
            logger.error("Cannot find experiment specification in config file!")
            raise ValueError

        if "missions" not in self.params["experiment"].keys():
            logger.error("Cannot find experiment missions specification in config file!")
            raise ValueError

        return self.params["experiment"]["missions"]

    def setup(self):
        grid_map = GridMap(self.params)

        sensor_model_factory = SensorModelFactory(self.params)
        sensor_model = sensor_model_factory.create_sensor_model()

        for i in range(self.repetitions):
            sensor_factory = SensorFactory(self.params, sensor_model, grid_map)
            sensor = sensor_factory.create_sensor()

            sensor_simulation_factory = SimulationFactory(self.params, sensor)
            sensor_simulation = sensor_simulation_factory.create_sensor_simulation()
            sensor.set_sensor_simulation(sensor_simulation)

            self.mappings[i] = Mapping(grid_map, sensor)

        for mission_params in self.missions_params:
            adapted_params = self.params.copy()
            adapted_params["mission"] = mission_params
            adapted_params["mission"].update(self.constraints_params)
            adapted_params["mission"].update(self.scenario_params)
            mission_type = f"{adapted_params['mission']['type']}_{adapted_params['mission']['config_name']}"
            self.missions[mission_type] = {"params": adapted_params, "instances": []}

    def run(self):
        for mission_type in self.missions.keys():
            for i in range(self.repetitions):
                mission_factory = MissionFactory(
                    self.missions[mission_type]["params"],
                    copy.deepcopy(self.mappings[i]),
                    self.use_effective_mission_time,
                )
                mission = mission_factory.create_mission()
                self.missions[mission_type]["instances"].append(mission)

                logger.info(
                    f"\n---------- EXECUTE {mission.mission_label} - {i + 1} of {self.repetitions} ----------\n"
                )
                mission.execute()

    def plot_performance_metric(
        self,
        metric: str,
        title: str,
        y_label: str,
        save_path: str = None,
        use_effective_mission_time: bool = False,
    ):
        total_performance_metrics_df = pd.DataFrame({"method": [], "flight_time": [], "metric": []})
        for mission_type in self.missions.keys():
            xs = []
            ys = []
            for i, mission in enumerate(self.missions[mission_type]["instances"]):
                metric_values = mission.map_uncertainties
                if metric == EvaluationMeasure.RMSE:
                    metric_values = mission.root_mean_squared_errors
                elif metric_values == EvaluationMeasure.WRMSE:
                    metric_values = mission.weighted_root_mean_squared_errors
                elif metric == EvaluationMeasure.MLL:
                    metric_values = mission.mean_log_losses
                elif metric == EvaluationMeasure.WMLL:
                    metric_values = mission.weighted_mean_log_losses
                elif metric == EvaluationMeasure.UNCERTAINTY_DIFFERENCE:
                    metric_values = mission.map_uncertainty_differences

                flight_times = np.array(mission.flight_times)
                if use_effective_mission_time:
                    flight_times += np.array(mission.run_times)
                flight_times = np.cumsum(flight_times)

                xs.append(flight_times)
                ys.append(np.array(metric_values))

            max_x = min(max([np.max(x) for x in xs]), self.constraints_params["budget"])
            mean_x_axis = np.linspace(0, max_x, int(np.ceil(max_x)))
            ys_interpolated = [np.interp(mean_x_axis, xs[i], ys[i]) for i in range(len(xs))]

            for y_interpolated in ys_interpolated:
                performance_metrics_df = pd.DataFrame({"flight_time": mean_x_axis, "metric": y_interpolated})
                performance_metrics_df["method"] = self.missions[mission_type]["instances"][0].mission_label
                total_performance_metrics_df = total_performance_metrics_df.append(
                    performance_metrics_df, ignore_index=True
                )

        ax = sns.lineplot(
            x="flight_time",
            y="metric",
            hue="method",
            data=total_performance_metrics_df,
            ci="sd",
            err_style="band",
            linewidth=PLOT_LINE_WIDTH,
            palette=self.get_color_palette(),
        )
        ax.set_ylabel(y_label, fontsize=PLOT_LABEL_FONT_SIZE)
        if use_effective_mission_time:
            ax.set_xlabel("Mission Time [s]", fontsize=PLOT_LABEL_FONT_SIZE)
        else:
            ax.set_xlabel("Flight Time [s]", fontsize=PLOT_LABEL_FONT_SIZE)

        ax.tick_params(labelsize=PLOT_TICKS_SIZE)

        plt.setp(ax.get_legend().get_texts(), fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.get_legend().set_title(None)

        if metric != EvaluationMeasure.UNCERTAINTY:
            ax.get_legend().remove()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_num_waypoints(self, save_path: str = None):
        waypoints_df = pd.DataFrame()
        for mission_type in self.missions.keys():
            num_waypoints = np.array(
                [len(self.missions[mission_type]["instances"][i].waypoints) for i in range(self.repetitions)]
            )
            waypoints_df[self.missions[mission_type]["instances"][0].mission_label] = num_waypoints

        waypoints_df = waypoints_df.reset_index().melt("index", var_name="method", value_name="num_waypoints")
        ax = sns.boxplot(
            x="method",
            y="num_waypoints",
            data=waypoints_df,
            hue="method",
            orient="v",
            linewidth=PLOT_LINE_WIDTH,
            palette=self.get_color_palette(),
        )
        ax.set_xlabel(None)
        ax.set_ylabel("number of waypoints", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.tick_params(labelsize=PLOT_TICKS_SIZE)

        plt.setp(ax.get_legend().get_texts(), fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.get_legend().set_title(None)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    @staticmethod
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        return colors.LinearSegmentedColormap.from_list(
            "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n))
        )

    def plot_paths(self, save_path: str = None):
        """Visualize missions paths with chosen waypoints, i.e. measurement positions"""
        ax = plt.axes(projection="3d")

        for mission_type in self.missions.keys():
            if self.missions[mission_type]["instances"][0].mission_type != MissionType.MCTS_ZERO:
                continue
            mission_color = self.missions[mission_type]["params"]["mission"]["color"]
            best_run = self.missions[mission_type]["instances"][0]
            for i in range(1, self.repetitions):
                current_run = self.missions[mission_type]["instances"][i]
                if current_run.map_uncertainties[-1] < best_run.map_uncertainties[-1]:
                    best_run = current_run

            num_waypoints = len(best_run.waypoints)
            color_map = cm.jet(np.linspace(0, 1, num_waypoints))
            for i in range(num_waypoints - 1):
                ax.plot3D(
                    best_run.waypoints[i : i + 2 :, 0],
                    best_run.waypoints[i : i + 2, 1],
                    best_run.waypoints[i : i + 2, 2],
                    "o-",
                    markersize=5,
                    label=best_run.mission_label,
                    color=color_map[i],
                    linewidth=PLOT_LINE_WIDTH,
                )

            x = np.array(list(range(0, best_run.mapping.grid_map.x_dim + 1))) * best_run.mapping.grid_map.resolution
            y = np.array(list(range(0, best_run.mapping.grid_map.y_dim + 1))) * best_run.mapping.grid_map.resolution
            Y, X = np.meshgrid(y, x)

            ground_truth_map = best_run.mapping.sensor.sensor_simulation.ground_truth_map
            padded_ground_truth_map = np.pad(ground_truth_map, 1, "edge")[1:, 1:]
            grid_map_colors = cm.viridis(padded_ground_truth_map)

            ax.plot_surface(Y, X, np.zeros_like(padded_ground_truth_map), facecolors=grid_map_colors)

        ax.set_xticks([0, 20, 40])
        ax.set_yticks([0, 20, 40])
        ax.set_zticks([5, 10, 15])
        ax.tick_params(labelsize=PLOT_TICKS_SIZE)

        if save_path is not None:
            ax.view_init(23, 25)
            plt.savefig(save_path, dpi=300)

        plt.show()

    @staticmethod
    def filter_outliers(data: np.array, interval_factor: float = 1) -> np.array:
        return data[np.abs(data - np.median(data)) < interval_factor * np.std(data)]

    def plot_run_time(self, save_path: str = None):
        """Visualize missions run time spent per measurement positions"""
        run_times_df = pd.DataFrame()
        for mission_type in self.missions.keys():
            if self.missions[mission_type]["instances"][0].mission_type in BASELINE_MISSION_TYPES:
                continue

            run_times = []
            for i in range(self.repetitions):
                filtered_run_times = self.filter_outliers(
                    np.array(self.missions[mission_type]["instances"][i].run_times[1:])
                )
                run_times.extend(filtered_run_times.tolist())

            run_times_df[self.missions[mission_type]["instances"][0].mission_label] = pd.Series(run_times)

        run_times_df = run_times_df.reset_index().melt("index", var_name="method", value_name="run_times")
        ax = sns.boxplot(
            x="method",
            y="run_times",
            data=run_times_df,
            hue="method",
            linewidth=PLOT_LINE_WIDTH,
            orient="v",
            palette=self.get_color_palette(),
        )
        ax.set_xlabel(None)

        ax.set_ylabel("Runtime [s]", fontsize=PLOT_LABEL_FONT_SIZE)
        ax.tick_params(labelsize=PLOT_LABEL_FONT_SIZE)

        plt.setp(ax.get_legend().get_texts(), fontsize=PLOT_LEGEND_FONT_SIZE)
        ax.get_legend().set_title(None)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_ablation_statistics(self, save_path: str = None, use_effective_mission_time: bool = False):
        """Creates table with KPIs for an ablation study"""
        total_performance_metrics_df = pd.DataFrame(
            {
                "method": [],
                "tr_25": [],
                "tr_50": [],
                "tr_75": [],
                "rmse_25": [],
                "rmse_55": [],
                "rmse_75": [],
                "run_time": [],
            }
        )
        for mission_type in self.missions.keys():
            xs = []
            ys_uncertainty = []
            ys_rmse = []
            for i, mission in enumerate(self.missions[mission_type]["instances"]):
                uncertainty_values = mission.map_uncertainties
                rmse_values = mission.root_mean_squared_errors

                flight_times = np.array(mission.flight_times)
                if use_effective_mission_time:
                    flight_times += np.array(mission.run_times)
                flight_times = np.cumsum(flight_times)

                xs.append(flight_times)
                ys_uncertainty.append(np.array(uncertainty_values))
                ys_rmse.append(np.array(rmse_values))

            max_x = min(max([np.max(x) for x in xs]), self.constraints_params["budget"])
            mean_x_axis = np.linspace(0, max_x, int(np.ceil(max_x)))
            ys_uncertainty_interpolated = [np.interp(mean_x_axis, xs[i], ys_uncertainty[i]) for i in range(len(xs))]
            ys_rmse_interpolated = [np.interp(mean_x_axis, xs[i], ys_rmse[i]) for i in range(len(xs))]

            tr_25s, tr_50s, tr_75s = [], [], []
            for uncertainty_interpolated in ys_uncertainty_interpolated:
                idx_25, idx_50, idx_75 = int(max_x / 4), int(max_x / 2), int(max_x * 3 / 4)
                tr_25, tr_50, tr_75 = (
                    uncertainty_interpolated[idx_25],
                    uncertainty_interpolated[idx_50],
                    uncertainty_interpolated[idx_75],
                )
                tr_25s.append(tr_25)
                tr_50s.append(tr_50)
                tr_75s.append(tr_75)

            rmse_25s, rmse_50s, rmse_75s = [], [], []
            for rmse_interpolated in ys_rmse_interpolated:
                idx_25, idx_50, idx_75 = int(max_x / 4), int(max_x / 2), int(max_x * 3 / 4)
                rmse_25, rmse_50, rmse_75 = (
                    rmse_interpolated[idx_25],
                    rmse_interpolated[idx_50],
                    rmse_interpolated[idx_75],
                )
                rmse_25s.append(rmse_25)
                rmse_50s.append(rmse_50)
                rmse_75s.append(rmse_75)

            tr_25_mean, tr_50_mean, tr_75_mean = (
                np.mean(np.array(tr_25s)),
                np.mean(np.array(tr_50s)),
                np.mean(np.array(tr_75s)),
            )
            rmse_25_mean, rmse_50_mean, rmse_75_mean = (
                np.mean(np.array(rmse_25s)),
                np.mean(np.array(rmse_50s)),
                np.mean(np.array(rmse_75s)),
            )

            run_times = []
            for i in range(self.repetitions):
                filtered_run_times = self.filter_outliers(
                    np.array(self.missions[mission_type]["instances"][i].run_times[1:])
                )
                run_times.extend(filtered_run_times.tolist())

            mean_run_time = np.mean(np.array(run_times))

            performance_metrics_df = pd.DataFrame(
                {
                    "method": [self.missions[mission_type]["instances"][0].mission_label],
                    "tr_25": [tr_25_mean],
                    "tr_50": [tr_50_mean],
                    "tr_75": [tr_75_mean],
                    "rmse_25": [rmse_25_mean],
                    "rmse_55": [rmse_50_mean],
                    "rmse_75": [rmse_75_mean],
                    "run_time": [mean_run_time],
                }
            )
            total_performance_metrics_df = total_performance_metrics_df.append(
                performance_metrics_df, ignore_index=True
            )

        if save_path is not None:
            total_performance_metrics_df.to_csv(save_path, sep=",", index=False)

    def eval(self):
        plots_folder = os.path.join(self.save_folder, "plots")
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        if EvaluationMeasure.NUM_WAYPOINTS in self.evaluation_metrics:
            self.plot_num_waypoints(save_path=os.path.join(plots_folder, "num_waypoints.png"))
        if EvaluationMeasure.PATHS in self.evaluation_metrics:
            self.plot_paths(save_path=os.path.join(plots_folder, "paths.png"))
        if EvaluationMeasure.RUN_TIME in self.evaluation_metrics:
            self.plot_run_time(save_path=os.path.join(plots_folder, "run_time.png"))
        if EvaluationMeasure.ABLATION_STATISTICS in self.evaluation_metrics:
            self.plot_ablation_statistics(save_path=os.path.join(plots_folder, "ablation_statistics.csv"))
        if EvaluationMeasure.UNCERTAINTY in self.evaluation_metrics:
            self.plot_performance_metric(
                EvaluationMeasure.UNCERTAINTY,
                "Uncertainty Reduction",
                "tr(P)",
                save_path=os.path.join(plots_folder, "uncertainty.png"),
                use_effective_mission_time=self.use_effective_mission_time,
            )
        if EvaluationMeasure.UNCERTAINTY_DIFFERENCE in self.evaluation_metrics:
            self.plot_performance_metric(
                EvaluationMeasure.UNCERTAINTY_DIFFERENCE,
                "Uncertainty Difference",
                "Mean Variance Difference",
                save_path=os.path.join(plots_folder, "uncertainty_difference.png"),
                use_effective_mission_time=self.use_effective_mission_time,
            )
        if EvaluationMeasure.RMSE in self.evaluation_metrics:
            self.plot_performance_metric(
                EvaluationMeasure.RMSE,
                "RMSE Reduction",
                "RMSE",
                save_path=os.path.join(plots_folder, "rmse.png"),
                use_effective_mission_time=self.use_effective_mission_time,
            )
        if EvaluationMeasure.WRMSE in self.evaluation_metrics:
            self.plot_performance_metric(
                EvaluationMeasure.WRMSE,
                "WRMSE Reduction",
                "WRMSE",
                save_path=os.path.join(plots_folder, "wrmse.png"),
                use_effective_mission_time=self.use_effective_mission_time,
            )
        if EvaluationMeasure.MLL in self.evaluation_metrics:
            self.plot_performance_metric(
                EvaluationMeasure.MLL,
                "MLL Reduction",
                "MLL",
                save_path=os.path.join(plots_folder, "mll.png"),
                use_effective_mission_time=self.use_effective_mission_time,
            )
        if EvaluationMeasure.WMLL in self.evaluation_metrics:
            self.plot_performance_metric(
                EvaluationMeasure.WMLL,
                "WMLL Reduction",
                "WMLL",
                save_path=os.path.join(plots_folder, "wmll.png"),
                use_effective_mission_time=self.use_effective_mission_time,
            )

    def save(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        experiment_dict = {"mappings": self.mappings, "missions": self.missions, "params": self.params}
        experiment_filepath = os.path.join(self.save_folder, "experiment.pkl")

        with open(experiment_filepath, "wb") as experiment_file:
            pickle.dump(experiment_dict, experiment_file)
