import _pickle as cPickle
import bz2
import os
from random import shuffle
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torch import nn

from constants import TRAIN_DATA_DIR


class ReplayBuffer:
    def __init__(
        self, self_play_iteration: int, window_size: int, batch_size: int = 32, num_augmented_samples: int = 0
    ):
        self.data_file_paths = self.get_data_file_paths(self_play_iteration, window_size)
        self.batch_size = batch_size
        self.num_augmented_samples = num_augmented_samples
        shuffle(self.data_file_paths)

    @property
    def sample_size(self) -> int:
        return max(1, int(self.batch_size / (self.num_augmented_samples + 1)))

    def step(self):
        pass

    def update(self, indices: np.array, priorities: np.array):
        pass

    @staticmethod
    def get_data_file_paths(self_play_iteration: int, window_size: int) -> List:
        data_file_paths = []
        window_start_idx = max(0, self_play_iteration - window_size + 1)
        data_dirs = [
            os.path.join(TRAIN_DATA_DIR, f"iter_{i}") for i in range(window_start_idx, self_play_iteration + 1)
        ]

        for data_dir in data_dirs:
            for sample_file_path in os.listdir(data_dir):
                data_file_paths.append(os.path.join(data_dir, sample_file_path))

        return data_file_paths

    @staticmethod
    def load_batch_from_disk(sample_file_paths: List) -> List:
        loaded_batch_data = []
        for sample_file_path in sample_file_paths:
            with bz2.BZ2File(sample_file_path, "rb") as file:
                input_feature_planes, policy, value, reward, valid_actions_msk = cPickle.load(file)
                loaded_batch_data.append((input_feature_planes, policy, value, reward, valid_actions_msk))

        return loaded_batch_data

    def augment_random_crop(
        self, states: Tuple, policies: Tuple, values: Tuple, rewards: Tuple, valid_actions_msk: Tuple
    ) -> np.array:
        states = torch.FloatTensor(np.array(states).astype(np.float64))

        policies = np.tile(np.array(policies), (self.num_augmented_samples + 1, 1))
        values = np.tile(np.array(values).astype(np.float64), self.num_augmented_samples + 1)
        rewards = np.tile(np.array(rewards).astype(np.float64), self.num_augmented_samples + 1)
        valid_actions_msk = np.tile(valid_actions_msk, (self.num_augmented_samples + 1, 1))

        w, h = states.shape[2:]
        augmented_states = [states.data.cpu().numpy()]
        for _ in range(self.num_augmented_samples):
            random_shift = nn.Sequential(nn.ReplicationPad2d(4), torchvision.transforms.RandomCrop((w, h)))
            augmented_states.append(random_shift(states).data.cpu().numpy())

        augmented_states = np.vstack(augmented_states)
        return augmented_states, policies, values, rewards, valid_actions_msk

    def sample(self) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        raise NotImplementedError("Replay buffer does not implement 'sample()' method!")

    def __len__(self):
        return len(self.data_file_paths)


class ExperienceReplayBuffer(ReplayBuffer):
    def __init__(
        self, self_play_iteration: int, window_size: int, batch_size: int = 32, num_augmented_samples: int = 0
    ):
        super().__init__(self_play_iteration, window_size, batch_size, num_augmented_samples)

    def sample(self) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        sample_indices = np.random.choice(len(self.data_file_paths), size=self.sample_size)
        sample_file_paths = [self.data_file_paths[i] for i in sample_indices]
        sample_data = self.load_batch_from_disk(sample_file_paths)
        states, policies, values, rewards, valid_actions_msk = list(zip(*[sample for sample in sample_data]))

        if self.num_augmented_samples > 0:
            states, policies, values, rewards, valid_actions_msk = self.augment_random_crop(
                states, policies, values, rewards, valid_actions_msk
            )

        return states, policies, values, rewards, valid_actions_msk, sample_indices, np.ones(len(states))


class PrioritizedExperienceReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        self_play_iteration: int,
        window_size: int,
        batch_size: int = 32,
        alpha: float = 0.75,
        beta0: float = 0.5,
        num_epochs: int = 3,
    ):
        super().__init__(self_play_iteration, window_size, batch_size)

        self.alpha = alpha
        self.beta0 = beta0
        self.beta = beta0
        self.priorities = np.ones(len(self.data_file_paths)) / len(self.data_file_paths)
        self.total_steps = (len(self.data_file_paths) // self.sample_size) * num_epochs

    def step(self):
        self.beta = np.minimum(self.beta + (1 - self.beta0) / self.total_steps, 1)

    def sample(self) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        probabilities = self.priorities ** self.alpha
        probabilities /= probabilities.sum()

        sample_indices = np.random.choice(len(self.data_file_paths), size=self.sample_size, p=probabilities)

        sample_file_paths = [self.data_file_paths[i] for i in sample_indices]
        sample_data = self.load_batch_from_disk(sample_file_paths)
        states, policies, values, rewards, valid_actions_msk = list(zip(*[sample for sample in sample_data]))

        weights = (probabilities[sample_indices] * len(self.data_file_paths)) ** (-self.beta)
        weights = np.array(weights / weights.max(), dtype=np.float32)

        return states, policies, values, rewards, valid_actions_msk, sample_indices, weights

    def update(self, indices: np.array, priorities: np.array):
        self.priorities[indices] = priorities
