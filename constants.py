import logging
import os

logger = logging.getLogger(__name__)


def load_from_env(env_var_name: str, data_type: callable, default=None):
    if env_var_name in os.environ and os.environ[env_var_name] != "":
        value = os.environ[env_var_name]
        if data_type == bool:
            if value.lower() == "true":
                value = True
            else:
                value = False
        else:
            value = data_type(value)
        return value
    elif env_var_name not in os.environ and default is None:
        raise ValueError(
            f"Could not find environment variable '{env_var_name}'. "
            f"Please check .env file or provide a default value when calling load_from_env()."
        )
    return default


PLOT_LABEL_FONT_SIZE = 30
PLOT_LEGEND_FONT_SIZE = 20
PLOT_TICKS_SIZE = 20
PLOT_LINE_WIDTH = 5

REPO_DIR = "/mapping_ipp_framework"

CONFIG_FILE_PATH = load_from_env("CONFIG_FILE_PATH", str, "config/example.yaml")
CONFIG_FILE_PATH = os.path.join(REPO_DIR, CONFIG_FILE_PATH)

CHECKPOINTS_DIR = load_from_env("CHECKPOINTS_DIR", str, "checkpoints")
CHECKPOINTS_DIR = os.path.join(REPO_DIR, CHECKPOINTS_DIR)

TRAIN_DATA_DIR = load_from_env("TRAIN_DATA_DIR", str, "generated_train_data")
TRAIN_DATA_DIR = os.path.join(REPO_DIR, TRAIN_DATA_DIR)

EXPERIMENTS_FOLDER = load_from_env("EXPERIMENT_FILE_PATH", str, "results")
EXPERIMENTS_FOLDER = os.path.join(REPO_DIR, EXPERIMENTS_FOLDER)

LOG_DIR = load_from_env("LOG_DIR", str, "logs")
LOG_DIR = os.path.join(REPO_DIR, LOG_DIR)
LOG_LEVEL = logging.DEBUG

TELEGRAM_CHAT_ID = load_from_env("TELEGRAM_CHAT_ID", str, "my_telegram_chat_id")
TELEGRAM_TOKEN = load_from_env("TELEGRAM_TOKEN", str, "my_telegram_token")

DATASETS_DIR = load_from_env("DATASETS_DIR", str, "datasets")
DATASETS_DIR = os.path.join(REPO_DIR, DATASETS_DIR)


class SensorType:
    RGB_CAMERA = "rgb_camera"


class SensorParams:
    CAMERA = ["field_of_view"]
    RGB_CAMERA = ["encoding"]


class SensorModelType:
    ALTITUDE_DEPENDENT = "altitude_dependent"


class SensorModelParams:
    ALTITUDE_DEPENDENT = ["coeff_a", "coeff_b"]


class SensorSimulationType:
    GAUSSIAN_RANDOM_FIELD = "gaussian_random_field"
    HOTSPOT_RANDOM_FIELD = "hotspot_random_field"
    SPLIT_RANDOM_FIELD = "split_random_field"
    TEMPERATURE_DATA_FIELD = "temperature_data_field"


class SensorSimulationParams:
    GAUSSIAN_RANDOM_FIELD = ["cluster_radius"]
    HOTSPOT_RANDOM_FIELD = ["cluster_radius"]
    SPLIT_RANDOM_FIELD = ["cluster_radius"]
    TEMPERATURE_DATA_FIELD = ["filename"]


SENSOR_TYPES = ["rgb_camera"]
SENSOR_MODELS = ["altitude_dependent"]
SENSOR_SIMULATIONS = ["gaussian_random_field", "hotspot_random_field", "split_random_field", "temperature_data_field"]


class MissionType:
    CONICAL_SPIRAL = "conical_spiral"
    LAWNMOWER = "lawnmower"
    RANDOM_CONTINUOUS = "random_continuous"
    RANDOM_DISCRETE = "random_discrete"
    GREEDY = "greedy"
    MCTS = "mcts"
    IPP_MASHA = "ipp_masha"
    MCTS_ZERO = "mcts_zero"


class MissionParams:
    STATIC_MISSION = [
        "dist_to_boundaries",
        "min_altitude",
        "max_altitude",
        "budget",
        "adaptive",
        "value_threshold",
        "interval_factor",
        "config_name",
    ]
    CONICAL_SPIRAL = ["num_waypoints", "slope_factor"]
    LAWNMOWER = ["step_size", "altitude_spacing"]
    RANDOM_CONTINUOUS = []
    RANDOM_DISCRETE = ["altitude_spacing"]
    GREEDY = ["num_waypoints", "altitude_spacing"]
    MCTS = [
        "altitude_spacing",
        "num_simulations",
        "gamma",
        "c",
        "episode_horizon",
        "k",
        "alpha",
        "epsilon_expand",
        "epsilon_rollout",
        "max_greedy_radius",
        "use_gcb_rollout",
    ]
    IPP_MASHA = [
        "episode_horizon",
        "altitude_spacing",
        "cmaes_max_iter",
        "cmaes_sigma0",
        "cmaes_population_size",
    ]
    MCTS_ZERO = [
        "altitude_spacing",
        "episode_horizon",
        "model_deployment_filename",
        "train_examples_iter",
        "restart_training",
        "telegram_notifications",
        {
            "hyper_params": [
                "gamma",
                "puct_init",
                "puct_init_decay",
                "puct_init_min",
                "puct_base",
                "forced_playout_factor",
                "num_mcts_simulations",
                "max_valid_action_distance",
                "max_episode_steps",
                "temperature_threshold",
                "num_self_play_iterations",
                "num_episodes",
                "start_train_examples_history",
                "train_examples_history_step",
                "max_train_examples_history",
                "num_arena_games",
                "network_update_threshold",
                "learning_rate",
                "max_learning_rate",
                "weight_decay",
                "num_epochs",
                "batch_size",
                "input_channels",
                "use_fov_input",
                "use_action_costs_input",
                "num_channels",
                "num_encoder_res_blocks",
                "num_policy_head_conv_bn_blocks",
                "num_value_head_conv_bn_blocks",
                "shared_network",
                "dropout",
                "max_grad_norm",
                "lr_step_size",
                "lr_decay",
                "policy_loss_coeff",
                "value_loss_coeff",
                "reward_loss_coeff",
                "reconstruction_loss_coeff",
                "entropy_regularization_coeff",
                "dirichlet_alpha",
                "dirichlet_alpha_decay",
                "dirichlet_alpha_min",
                "dirichlet_eps",
                "continuous_network_update",
                "reset_mcts_each_step",
                "momentum",
                "temperature_scale",
                "shuffle_train_env_intervals",
                "shuffle_budget",
                "shuffle_prior_cov",
                "num_workers",
                "max_inference_batch_size",
                "max_waiting_time",
                "non_blocking_read",
                "use_autoencoder",
                "use_reward_target",
                "replay_alpha",
                "replay_beta0",
                "use_per",
                "mask_policy_head",
                "use_silu",
                "use_separable_conv_layers",
                "num_augmented_samples",
                "input_history_length",
                "log_network_parameters",
                "use_global_context_mixing",
                "num_global_pooling_channels",
            ]
        },
    ]


BASELINE_MISSION_TYPES = ["conical_spiral", "lawnmower", "random_continuous", "random_discrete"]
MISSION_TYPES = BASELINE_MISSION_TYPES + [
    "greedy",
    "mcts",
    "ipp_masha",
    "mcts_zero",
]

UAV_PARAMS = ["max_v", "max_a", "sampling_time"]


class EvaluationMeasure:
    NUM_WAYPOINTS = "num_waypoints"
    PATHS = "paths"
    UNCERTAINTY = "uncertainty"
    UNCERTAINTY_DIFFERENCE = "uncertainty_difference"
    RMSE = "rmse"
    WRMSE = "wrmse"
    MLL = "mll"
    WMLL = "wmll"
    RUN_TIME = "run_time"
    ABLATION_STATISTICS = "ablation_statistics"


def log_env_variables():
    env_variables = {
        "REPO_DIR": REPO_DIR,
        "CONFIG_FILE_PATH": CONFIG_FILE_PATH,
        "CHECKPOINTS_DIR": CHECKPOINTS_DIR,
        "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
        "EXPERIMENTS_FOLDER": EXPERIMENTS_FOLDER,
        "LOG_DIR": LOG_DIR,
    }

    logger.info("\n-------------------------------------- LOG ENV-VARIABLES --------------------------------------\n")
    for env_var in env_variables.keys():
        logger.info(f"{env_var}: {env_variables[env_var]} | type: {type(env_variables[env_var])}")
