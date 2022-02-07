import logging
import os
import time
from multiprocessing import Queue
from typing import Dict, Optional, Tuple, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from constants import LOG_DIR
from planning.mcts_zero.network_wrappers.policy_network_wrappers import PolicyNetworkWrapper
from planning.mcts_zero.network_wrappers.policy_value_network_wrappers import PolicyValueNetworkWrapper
from planning.mcts_zero.network_wrappers.value_network_wrappers import ValueNetworkWrapper

logger = logging.getLogger(__name__)


def load_network(
    hyper_params: Dict, meta_data: Dict, network_filename: str, value_network_filename: str = None
) -> Tuple[Union[PolicyValueNetworkWrapper, PolicyNetworkWrapper], Optional[ValueNetworkWrapper]]:
    if value_network_filename is None:
        network = PolicyValueNetworkWrapper(hyper_params, meta_data)
        network.set_summary_writer(SummaryWriter(os.path.join(LOG_DIR, "tensorboard")))
        network.load_checkpoint(filename=network_filename)
        return network, None

    policy_network = PolicyNetworkWrapper(hyper_params, meta_data)
    policy_network.set_summary_writer(SummaryWriter(os.path.join(LOG_DIR, "tensorboard")))
    policy_network.load_checkpoint(filename=network_filename)

    value_network = ValueNetworkWrapper(hyper_params)
    value_network.set_summary_writer(SummaryWriter(os.path.join(LOG_DIR, "tensorboard")))
    value_network.load_checkpoint(filename=value_network_filename)

    return policy_network, value_network


def inference_worker(
    process_index: int,
    network_queue: Queue,
    request_queue: Queue,
    reply_queues: Dict,
    hyper_params: Dict,
    meta_data: Dict,
    network_filename: str,
    max_batch_size: int,
    max_waiting_time: float,
    value_network_filename: str = None,
    log_network_parameters: bool = False,
):
    inference_counter = 0
    request_data = {}
    request_order = []
    network, value_network = load_network(hyper_params, meta_data, network_filename, value_network_filename)
    last_inference = time.time()

    def _track_inference_statistics():
        network.writer.add_scalar(f"SelfPlay/ValuePredicted", value, time.time())
        network.writer.add_scalar(f"SelfPlay/InferenceRuns", inference_counter, time.time())
        if log_network_parameters:
            network.writer.add_histogram(f"SelfPlay/PolicyPredicted", policy, time.time())

    while True:
        if not network_queue.empty():
            network_msg = network_queue.get()
            if network_msg == "END":
                break

            if network_msg == "LOAD":
                network, value_network = load_network(hyper_params, meta_data, network_filename, value_network_filename)
                logger.info(f"Inference worker loaded new networks")

        max_waiting_time_exceeded = (
            0 < len(request_order) < max_batch_size and ((time.time() - last_inference) * 1000) > max_waiting_time
        )
        if max_waiting_time_exceeded:
            inference_counter += 1
            batch_data = [data[0] for data in request_data.values()]
            action_msk = [data[1] for data in request_data.values()]
            if value_network is None:
                policies, values = network.predict(np.asarray(batch_data), np.array(action_msk))
            else:
                policies = network.predict(np.asarray(batch_data), np.array(action_msk))
                values = value_network.predict(np.asarray(batch_data))

            for i in range(len(values)):
                policy, value = policies[i], values[i]
                reply_id = request_order[i]
                response_msg = {"policy": policy, "value": value}
                reply_queues[reply_id].put(response_msg)

                _track_inference_statistics()

            request_data = {}
            request_order = []
            last_inference = time.time()

        if request_queue.empty():
            continue

        request_msg = request_queue.get_nowait()
        if "id" in request_msg and "input" in request_msg and "action_msk" in request_msg:
            worker_id = request_msg["id"]
            request_data[worker_id] = (request_msg["input"], request_msg["action_msk"])
            request_order.append(worker_id)

        if len(request_order) == max_batch_size:
            inference_counter += 1
            batch_data = [data[0] for data in request_data.values()]
            action_msk = [data[1] for data in request_data.values()]
            if value_network is None:
                policies, values = network.predict(np.asarray(batch_data), np.array(action_msk))
            else:
                policies = network.predict(np.asarray(batch_data), np.array(action_msk))
                values = value_network.predict(np.asarray(batch_data))

            for i in range(len(values)):
                policy, value = policies[i], values[i]
                reply_id = request_order[i]
                response_msg = {"policy": policy, "value": value}
                reply_queues[reply_id].put(response_msg)

                _track_inference_statistics()

            request_data = {}
            request_order = []
            last_inference = time.time()
