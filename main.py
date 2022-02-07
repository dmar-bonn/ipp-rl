import constants
from config.params import load_params
from experiments.experiments import Experiment
from logger import setup_logger


def main():
    constants.log_env_variables()
    params = load_params(constants.CONFIG_FILE_PATH)

    logger.info("\n-------------------------------------- START PIPELINE --------------------------------------\n")

    experiment = Experiment(params)
    experiment.run()
    experiment.eval()
    experiment.save()

    logger.info("\n-------------------------------------- STOP PIPELINE --------------------------------------\n")


if __name__ == "__main__":
    logger = setup_logger()
    main()
