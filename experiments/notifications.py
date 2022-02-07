import logging
from typing import Dict

from telegram import Bot

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, mission_label: str, token: str, chat_id: str, hyper_params: Dict = None, verbose: bool = True):
        self.chat_id = chat_id
        self.bot = Bot(token)
        self.verbose = verbose
        self.mission_label = mission_label
        self.hyper_params = hyper_params

    def start_experiment(self):
        """Record the start of the experiment and send telegram message."""
        start_text = f"Start {self.mission_label} mission!\n\n"
        if self.verbose:
            start_text += f"Mission's parameters:\n"
            start_text += self.format_info_dict(self.hyper_params)

        self.send_message(start_text)

    def finished_iteration(self, iteration_id: str, additional_info: Dict = None):
        """Record the completion of an iteration and send telegram message."""
        iteration_text = f"Completed {iteration_id} iteration of {self.mission_label} experiment!\n\n"

        if self.verbose:
            iteration_text += "Additional Iteration Info:\n"
            iteration_text += self.format_info_dict(additional_info)

        self.send_message(iteration_text)

    def finish_experiment(self):
        """Record the termination of the experiment. Summarize completed and failed runs."""
        finish_text = f"Finished {self.mission_label} experiment!\n\n"
        if self.verbose:
            finish_text += f"Mission's parameters:\n{self.hyper_params}"

        self.send_message(finish_text)

    def failed_experiment(self, e: Exception):
        failed_text = f"Experiment {self.mission_label} failed with the following exception:\n{e}"
        self.send_message(failed_text)

    def send_message(self, message):
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
        self.bot.send_message(chat_id=self.chat_id, text=message)

    @staticmethod
    def format_info_dict(info_dict: Dict) -> str:
        if info_dict is None:
            return "No further information given"

        info_str = ""
        for key, value in info_dict.items():
            info_str += f"{key}: {value}\n"

        return info_str.strip()
