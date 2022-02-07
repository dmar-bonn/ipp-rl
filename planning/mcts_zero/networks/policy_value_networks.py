import logging
from typing import Dict

import torch.cuda
from torch import nn

from planning.common.layers import Decoder, Encoder, PolicyHead, ValueHead

logger = logging.getLogger(__name__)


class PolicyValueNetwork(nn.Module):
    def __init__(self, hyper_params: Dict, meta_data: Dict):
        num_altitude_levels = (
            int((meta_data["max_altitude"] - meta_data["min_altitude"]) / meta_data["altitude_spacing"]) + 1
        )
        self.num_actions = meta_data["num_grid_cells"] * num_altitude_levels
        self.hyper_params = hyper_params

        super(PolicyValueNetwork, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_channels = self.hyper_params["input_channels"]
        num_channels = self.hyper_params["num_channels"]
        dropout = self.hyper_params["dropout"]
        nonlinearity = nn.SiLU() if self.hyper_params["use_silu"] else nn.ReLU()

        self.encoder = Encoder(
            input_channels,
            num_channels,
            nonlinearity,
            self.hyper_params["num_encoder_res_blocks"],
            dropout=dropout,
            use_separable_conv_layers=self.hyper_params["use_separable_conv_layers"],
            use_global_context_mixing=self.hyper_params["use_global_context_mixing"],
            num_global_pooling_channels=self.hyper_params["num_global_pooling_channels"],
        )

        self.policy_head = PolicyHead(
            num_channels,
            self.hyper_params["num_policy_head_conv_bn_blocks"],
            nonlinearity,
            self.num_actions,
            dropout=dropout,
            mask_policy_head=self.hyper_params["mask_policy_head"],
            use_global_context_mixing=self.hyper_params["use_global_context_mixing"],
            num_global_pooling_channels=self.hyper_params["num_global_pooling_channels"],
        )

        self.value_head = ValueHead(
            num_channels,
            self.hyper_params["num_value_head_conv_bn_blocks"],
            nonlinearity,
            dropout=dropout,
            use_reward_target=self.hyper_params["use_reward_target"],
            use_global_context_mixing=self.hyper_params["use_global_context_mixing"],
            num_global_pooling_channels=self.hyper_params["num_global_pooling_channels"],
        )
        self.decoder = Decoder(num_channels, nonlinearity, dropout=dropout)

    def forward(self, x, valid_actions_msk):
        x = self.encoder(x)

        x_policy = self.policy_head(x, valid_actions_msk)
        x_value, x_reward = self.value_head(x)
        x_decoder = self.decoder(x) if self.hyper_params["use_autoencoder"] else None

        return x_policy, x_value, x_reward, x_decoder
