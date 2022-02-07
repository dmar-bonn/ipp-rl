import logging
from typing import Dict

import torch.cuda
from torch import nn

from planning.common.layers import Decoder, Encoder, ValueHead

logger = logging.getLogger(__name__)


class ValueNetwork(nn.Module):
    def __init__(self, hyper_params: Dict):
        super(ValueNetwork, self).__init__()

        self.hyper_params = hyper_params
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

    def forward(self, x):
        x = self.encoder(x)

        x_value, x_reward = self.value_head(x)
        x_decoder = self.decoder(x) if self.hyper_params["use_autoencoder"] else None

        return x_value, x_reward, x_decoder
