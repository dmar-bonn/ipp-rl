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
from planning.mcts_zero.networks.policy_value_networks import PolicyValueNetwork
from planning.mcts_zero.replay_buffers import ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer

logger = logging.getLogger(__name__)


class PolicyValueNetworkWrapper:
    def __init__(self, hyper_params: Dict, meta_data: Dict):
        self.meta_data = meta_data
        self.hyper_params = hyper_params
        self.network = PolicyValueNetwork(self.hyper_params, self.meta_data)
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
            policy_losses = AverageMeter()
            value_losses = AverageMeter()
            reward_losses = AverageMeter()
            reconstruction_losses = AverageMeter()
            entropy_regs = AverageMeter()

            num_batches = len(replay_buffer.data_file_paths) // replay_buffer.sample_size
            t = tqdm(range(num_batches), desc="Train network")
            for _ in t:
                (
                    states_tmp,
                    policies_tmp,
                    values_tmp,
                    rewards_tmp,
                    valid_actions_msk_tmp,
                    indices,
                    weights_tmp,
                ) = replay_buffer.sample()
                states_np, policies_np, values_np, rewards_np = (
                    np.array(states_tmp).astype(np.float64),
                    np.array(policies_tmp),
                    np.array(values_tmp).astype(np.float64),
                    np.array(rewards_tmp).astype(np.float64),
                )

                states = torch.FloatTensor(states_np)
                target_policies = torch.FloatTensor(policies_np)
                target_values = torch.FloatTensor(values_np)
                target_rewards = torch.FloatTensor(rewards_np)
                valid_actions_msk = torch.FloatTensor(valid_actions_msk_tmp)
                weights = torch.FloatTensor(weights_tmp)

                del states_tmp, policies_tmp, values_tmp, rewards_tmp, valid_actions_msk_tmp, weights_tmp
                del states_np, policies_np, values_np, rewards_np
                gc.collect()

                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)
                target_rewards = target_rewards.to(self.device)
                weights = weights.to(self.device)
                valid_actions_msk = valid_actions_msk.to(self.device)

                predicted_policies, predicted_values, predicted_rewards, states_reconstruction = self.network(
                    states, valid_actions_msk
                )

                policy_losses_iter = self.policy_loss(target_policies, predicted_policies, valid_actions_msk)
                value_losses_iter = self.scalar_loss(target_values, predicted_values)
                total_losses_iter = (
                    self.hyper_params["policy_loss_coeff"] * policy_losses_iter
                    + self.hyper_params["value_loss_coeff"] * value_losses_iter
                )

                entropy_regs_iter = self.entropy_regularization(predicted_policies)
                total_losses_iter -= self.hyper_params["entropy_regularization_coeff"] * entropy_regs_iter

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
                    self.writer.add_scalar("Loss/Reconstruction/Train", reconstruction_loss_iter, self.total_iterations)
                    reconstruction_losses.update(reconstruction_loss_iter.item(), states.size()[0])

                total_losses_iter *= weights

                entropy_reg_iter = entropy_regs_iter.mean()
                value_loss_iter = value_losses_iter.mean()
                policy_loss_iter = policy_losses_iter.mean()
                total_loss_iter = total_losses_iter.mean()

                entropy_regs.update(entropy_reg_iter.item(), states.size()[0])
                policy_losses.update(policy_loss_iter.item(), states.size()[0])
                value_losses.update(value_loss_iter.item(), states.size()[0])
                t.set_postfix(
                    policy_loss=policy_losses,
                    value_loss=value_losses,
                    reward_loss=reward_losses,
                    reconstruction_loss=reconstruction_losses,
                    entropy=entropy_regs,
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

                self.writer.add_scalar("HyperParams/LearningRate", scheduler.get_last_lr()[0], self.total_iterations)
                self.writer.add_scalar("Loss/Policy/Train", policy_loss_iter, self.total_iterations)
                self.writer.add_scalar("Loss/Value/Train", value_loss_iter, self.total_iterations)
                self.writer.add_scalar(
                    "Loss/ValueRelative/Train",
                    self.relative_scalar_loss(target_values, predicted_values),
                    self.total_iterations,
                )
                self.writer.add_scalar("Loss/Total/Train", total_loss_iter, self.total_iterations)
                self.writer.add_scalar("Loss/Entropy/Train", entropy_reg_iter, self.total_iterations)
                if self.hyper_params["use_per"]:
                    self.writer.add_scalar("HyperParams/ReplayBeta", replay_buffer.beta, self.total_iterations)

                total_grad_norm = 0
                for params in self.network.parameters():
                    if params.grad is not None:
                        total_grad_norm += params.grad.data.norm(2).item()

                self.writer.add_scalar(f"Gradients/TotalNorm", total_grad_norm, self.total_iterations)
                self.total_iterations += 1

                del states, target_policies, target_values, target_rewards, weights, valid_actions_msk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.hyper_params["log_network_parameters"]:
                for tag, params in self.network.named_parameters():
                    if params.grad is not None:
                        self.writer.add_histogram(f"Parameters/{tag}", params.data.cpu().numpy(), self.total_iterations)

                for tag, params in self.network.named_parameters():
                    if params.grad is not None:
                        self.writer.add_histogram(
                            f"Gradients/{tag}", params.grad.data.cpu().numpy(), self.total_iterations
                        )

        del optimizer, replay_buffer, scheduler
        gc.collect()

    def predict(self, state, valid_actions_msk):
        state = torch.FloatTensor(state.astype(np.float64))
        valid_actions_msk = torch.FloatTensor(valid_actions_msk)
        state = state.to(self.device)
        valid_actions_msk = valid_actions_msk.to(self.device)

        self.network.eval()

        with torch.no_grad():
            policy, value, _, _ = self.network(state, valid_actions_msk)

        value = value.data.cpu().numpy().flatten()
        value = invert_scaled_value_target(value)

        return torch.exp(policy).data.cpu().numpy(), value

    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        if not os.path.exists(CHECKPOINTS_DIR):
            logger.info(f"Create directory {CHECKPOINTS_DIR}")
            os.mkdir(CHECKPOINTS_DIR)

        filepath = os.path.join(CHECKPOINTS_DIR, filename)
        torch.save({"state_dict": self.network.state_dict()}, filepath)

    def load_checkpoint(self, filename="checkpoint.pth.tar"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(CHECKPOINTS_DIR, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path {filepath}")

        map_location = None if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.network.load_state_dict(checkpoint["state_dict"])

    @staticmethod
    def entropy_regularization(predicted):
        return -torch.sum(torch.exp(predicted) * predicted, dim=1)

    @staticmethod
    def policy_loss(targets, predicted, valid_actions_msk):
        return -torch.sum(targets * predicted * valid_actions_msk, dim=1)

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
