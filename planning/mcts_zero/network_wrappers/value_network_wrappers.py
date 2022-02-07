import gc
import logging
import os
from typing import Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants import CHECKPOINTS_DIR
from planning.common.rewards import invert_scaled_value_target
from planning.common.statistics import AverageMeter
from planning.mcts_zero.networks.value_networks import ValueNetwork
from planning.mcts_zero.replay_buffers import ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer

logger = logging.getLogger(__name__)


class ValueNetworkWrapper:
    def __init__(self, hyper_params: Dict):
        self.hyper_params = hyper_params
        self.network = ValueNetwork(self.hyper_params)
        self.total_iterations = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = self.network.to(self.device)
        self.writer = None

    def set_summary_writer(self, writer: SummaryWriter):
        self.writer = writer

    def train(self, self_play_iteration: int, window_size: int):
        replay_buffer = ExperienceReplayBuffer(
            self_play_iteration,
            window_size,
            self.hyper_params["batch_size"],
            self.hyper_params["num_augmented_samples"],
        )
        if self.hyper_params["use_per"]:
            replay_buffer = PrioritizedExperienceReplayBuffer(
                self_play_iteration,
                window_size,
                self.hyper_params["batch_size"],
                self.hyper_params["replay_alpha"],
                self.hyper_params["replay_beta0"],
                self.hyper_params["num_epochs"],
            )

        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hyper_params["learning_rate"],
            weight_decay=self.hyper_params["weight_decay"],
            momentum=self.hyper_params["momentum"],
        )

        num_batches = len(replay_buffer.data_file_paths) // replay_buffer.sample_size
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hyper_params["max_learning_rate"],
            epochs=self.hyper_params["num_epochs"],
            steps_per_epoch=num_batches,
            div_factor=self.hyper_params["max_learning_rate"] / self.hyper_params["learning_rate"],
            final_div_factor=100,
            anneal_strategy="linear",
            three_phase=True,
            pct_start=0.40,
        )
        for epoch in range(self.hyper_params["num_epochs"]):
            logger.info(f"EPOCH: {epoch}")
            self.network.train()
            value_losses = AverageMeter()
            reward_losses = AverageMeter()
            reconstruction_losses = AverageMeter()

            num_batches = len(replay_buffer.data_file_paths) // replay_buffer.sample_size
            t = tqdm(range(num_batches), desc="Train network")
            for _ in t:
                states_tmp, _, values_tmp, rewards_tmp, _, indices, weights_tmp = replay_buffer.sample()
                states_np, values_np, rewards_np = (
                    np.array(states_tmp).astype(np.float64),
                    np.array(values_tmp).astype(np.float64),
                    np.array(rewards_tmp).astype(np.float64),
                )

                states = torch.FloatTensor(states_np)
                target_values = torch.FloatTensor(values_np)
                target_rewards = torch.FloatTensor(rewards_np)
                weights = torch.FloatTensor(weights_tmp)

                del states_tmp, values_tmp, rewards_tmp, weights_tmp
                del states_np, values_np, rewards_np
                gc.collect()

                states = states.to(self.device)
                target_values = target_values.to(self.device)
                target_rewards = target_rewards.to(self.device)
                weights = weights.to(self.device)

                predicted_values, predicted_rewards, states_reconstruction = self.network(states)
                value_losses_iter = self.scalar_loss(target_values, predicted_values)
                total_losses_iter = self.hyper_params["value_loss_coeff"] * value_losses_iter

                if self.hyper_params["use_reward_target"]:
                    reward_losses_iter = self.scalar_loss(target_rewards, predicted_rewards)
                    total_losses_iter += self.hyper_params["reward_loss_coeff"] * reward_losses_iter
                    reward_loss_iter = reward_losses_iter.mean()
                    reward_losses.update(reward_loss_iter.item(), states.size()[0])
                    self.writer.add_scalar("Loss/Reward/Train", reward_loss_iter, self.total_iterations)
                    self.writer.add_scalar(
                        "Loss/RewardRelative/Train",
                        self.relative_scalar_loss(target_rewards, predicted_rewards),
                        self.total_iterations,
                    )

                if self.hyper_params["use_autoencoder"]:
                    reconstruction_losses_iter = self.reconstruction_loss(states[:, 0, :, :], states_reconstruction)
                    total_losses_iter += self.hyper_params["reconstruction_loss_coeff"] * reconstruction_losses_iter
                    reconstruction_loss_iter = reconstruction_losses_iter.mean()
                    self.writer.add_scalar(
                        "Loss/ValueNet/Reconstruction/Train", reconstruction_loss_iter, self.total_iterations
                    )
                    reconstruction_losses.update(reconstruction_loss_iter.item(), states.size()[0])

                total_losses_iter *= weights

                value_loss_iter = value_losses_iter.mean()
                total_loss_iter = total_losses_iter.mean()

                value_losses.update(value_loss_iter.item(), states.size()[0])
                t.set_postfix(
                    value_loss=value_losses, reward_loss=reward_losses, reconstruction_loss=reconstruction_losses
                )

                optimizer.zero_grad()
                total_loss_iter.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), max_norm=self.hyper_params["max_grad_norm"], norm_type=2
                )
                optimizer.step()
                scheduler.step()
                replay_buffer.step()
                replay_buffer.update(indices, (value_losses_iter + 1e-8).data.cpu())

                self.writer.add_scalar(
                    "HyperParams/ValueNet/LearningRate", scheduler.get_last_lr()[0], self.total_iterations
                )
                self.writer.add_scalar("Loss/Value/Train", value_loss_iter, self.total_iterations)
                self.writer.add_scalar(
                    "Loss/ValueRelative/Train",
                    self.relative_scalar_loss(target_values, predicted_values),
                    self.total_iterations,
                )
                self.writer.add_scalar("Loss/ValueNet/Total/Train", total_loss_iter, self.total_iterations)
                if self.hyper_params["use_per"]:
                    self.writer.add_scalar("HyperParams/ReplayBeta", replay_buffer.beta, self.total_iterations)

                total_grad_norm = 0
                for params in self.network.parameters():
                    if params.grad is not None:
                        total_grad_norm += params.grad.data.norm(2).item()

                self.writer.add_scalar(f"Gradients/ValueNet/TotalNorm", total_grad_norm, self.total_iterations)
                self.total_iterations += 1

                del states, target_values, target_rewards, weights
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.hyper_params["log_network_parameters"]:
                for tag, params in self.network.named_parameters():
                    if params.grad is not None:
                        self.writer.add_histogram(
                            f"Parameters/ValueNet/{tag}", params.data.cpu().numpy(), self.total_iterations
                        )

                for tag, params in self.network.named_parameters():
                    if params.grad is not None:
                        self.writer.add_histogram(
                            f"Gradients/ValueNet/{tag}", params.grad.data.cpu().numpy(), self.total_iterations
                        )

        del optimizer, replay_buffer, scheduler
        gc.collect()

    def predict(self, state):
        state = torch.FloatTensor(state.astype(np.float64))
        state = state.to(self.device)

        self.network.eval()

        with torch.no_grad():
            value, _, _ = self.network(state)

        value = value.data.cpu().numpy().flatten()
        value = invert_scaled_value_target(value)

        return value

    def save_checkpoint(self, filename="value_net.checkpoint.pth.tar"):
        if not os.path.exists(CHECKPOINTS_DIR):
            logger.info(f"Create directory {CHECKPOINTS_DIR}")
            os.mkdir(CHECKPOINTS_DIR)

        filepath = os.path.join(CHECKPOINTS_DIR, filename)
        torch.save({"state_dict": self.network.state_dict()}, filepath)

    def load_checkpoint(self, filename="value_net.checkpoint.pth.tar"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(CHECKPOINTS_DIR, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path {filepath}")

        map_location = None if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.network.load_state_dict(checkpoint["state_dict"])

    @staticmethod
    def scalar_loss(targets, predicted):
        return torch.square(predicted.view(-1) - targets)

    @staticmethod
    def relative_scalar_loss(targets, predicted):
        return torch.mean(torch.abs(targets - predicted.view(-1)) / torch.abs(targets))

    @staticmethod
    def reconstruction_loss(states, states_reconstruction):
        stated_reshaped = states.view(states.size()[0], np.prod(states.size()[1:]))
        states_reconstruction_reshaped = states_reconstruction.view(states.size()[0], np.prod(states.size()[1:]))
        loss = torch.sum((stated_reshaped - states_reconstruction_reshaped).pow(2), dim=1) / np.prod(states.size()[1:])
        return loss
