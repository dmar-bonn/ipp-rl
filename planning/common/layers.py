import torch
from torch import nn


def conv_bn(in_channels: int, out_channels: int, kernel_size: int, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs), nn.BatchNorm2d(out_channels)
    )


class ResidualBlock(nn.Module):
    def __init__(
        self, num_channels: int, dropout: float, stride: int, nonlinearity: callable, use_1x1conv: bool = False
    ):
        super().__init__()

        self.nonlinearity = nonlinearity
        self.num_channels = num_channels
        self.stride = stride
        self.use_1x1conv = use_1x1conv

        self.dropout = nn.Dropout2d(dropout)
        self.intermediate_block = nn.Sequential(
            conv_bn(self.num_channels, self.num_channels, kernel_size=3, stride=stride, padding=1),
            self.nonlinearity,
            conv_bn(self.num_channels, self.num_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv_identity = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x_identity = self.conv_identity(x) if self.use_1x1conv else x
        output = self.intermediate_block(x)

        if self.dropout.p > 0:
            output = self.dropout(output)

        return self.nonlinearity(output + x_identity)


class NonBottleneck1d(nn.Module):
    """Adapted from https://github.com/Eromera/erfnet_pytorch/blob/master/imagenet/erfnet_imagenet.py#L24-L57."""

    def __init__(
        self,
        num_channels: int,
        dropout: float,
        dilated: int,
        nonlinearity: callable,
        use_1x1conv: bool = False,
        down_sample: bool = False,
    ):
        super().__init__()

        self.down_sample = down_sample
        self.down_sample_layer = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_channels),
            nonlinearity,
        )

        self.use_1x1conv = use_1x1conv
        self.conv_identity = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1)

        self.conv3x1_1 = nn.Conv2d(num_channels, num_channels, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(num_channels, num_channels, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.conv3x1_2 = nn.Conv2d(
            num_channels, num_channels, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True, dilation=(dilated, 1)
        )
        self.conv1x3_2 = nn.Conv2d(
            num_channels, num_channels, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True, dilation=(1, dilated)
        )

        self.bn1 = nn.BatchNorm2d(num_channels, eps=1e-03)
        self.bn2 = nn.BatchNorm2d(num_channels, eps=1e-03)

        self.dropout = nn.Dropout2d(dropout)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if self.down_sample:
            x = self.down_sample_layer(x)

        x = self.conv_identity(x) if self.use_1x1conv else x

        output = self.conv3x1_1(x)
        output = self.nonlinearity(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.nonlinearity(output)

        output = self.conv3x1_2(output)
        output = self.nonlinearity(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p > 0:
            output = self.dropout(output)

        return self.nonlinearity(output + x)


class MixGlobalContext(nn.Module):
    def __init__(
        self,
        num_channels: int,
        nonlinearity: callable,
        dropout: float,
        num_global_pooling_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.stride = stride
        self.nonlinearity = nonlinearity
        self.num_global_pooling_channels = num_global_pooling_channels

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, bias=False, stride=self.stride, padding=1)
        self.conv_bn_layer_s1 = conv_bn(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.conv_identity = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=self.stride)

        self.global_pooling = GlobalPooling()
        self.fc_layer = nn.Sequential(
            nn.Linear(2 * num_global_pooling_channels, num_channels - num_global_pooling_channels), self.nonlinearity
        )
        self.bn_layer = nn.BatchNorm2d(self.num_global_pooling_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x_identity = self.conv_identity(x) if self.stride > 1 else x

        output = self.conv1(x)

        output_pool = output[:, : self.num_global_pooling_channels, :, :].clone()
        output_pool = self.bn_layer(output_pool)
        output_pool = self.nonlinearity(output_pool)
        output_pool = self.global_pooling(output_pool)
        output_pool = self.fc_layer(output_pool)

        n, c = output_pool.size()
        output_pool = output_pool.reshape([n, c, 1, 1])
        output[:, self.num_global_pooling_channels :, :, :] += output_pool

        output = self.conv_bn_layer_s1(output)
        if self.dropout.p > 0:
            output = self.dropout(output)

        return self.nonlinearity(output + x_identity)


class GlobalPooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.global_avg_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.global_max_pooling = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten())

    def forward(self, x):
        x_avg = self.global_avg_pooling(x)
        x_max = self.global_max_pooling(x)
        return torch.cat([x_avg, x_max], dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_channels: int,
        nonlinearity: callable,
        num_encoder_res_blocks: int,
        dropout: float = 0,
        use_separable_conv_layers: bool = False,
        use_global_context_mixing: bool = True,
        num_global_pooling_channels: int = 32,
    ):
        super().__init__()

        self.num_encoder_res_blocks = num_encoder_res_blocks
        self.use_separable_conv_layers = use_separable_conv_layers

        self.down_sample_block = nn.Sequential(
            conv_bn(input_channels, num_channels, kernel_size=7, stride=2, padding=3), nonlinearity,
        )

        self.residual_block_s1 = ResidualBlock(num_channels, dropout, 1, nonlinearity, use_1x1conv=True)
        self.residual_block_s2 = ResidualBlock(num_channels, dropout, 2, nonlinearity, use_1x1conv=True)

        self.separable_residual_block_s1 = NonBottleneck1d(
            num_channels, dropout, 1, nonlinearity, use_1x1conv=True, down_sample=False
        )
        self.separable_residual_block_s2 = NonBottleneck1d(
            num_channels, dropout, 1, nonlinearity, use_1x1conv=True, down_sample=True,
        )

        self.use_global_context_mixing = use_global_context_mixing
        self.mix_global_context_s1 = MixGlobalContext(
            num_channels, nonlinearity, dropout, num_global_pooling_channels, stride=1
        )
        self.mix_global_context_s2 = MixGlobalContext(
            num_channels, nonlinearity, dropout, num_global_pooling_channels, stride=2
        )

    def forward(self, x):
        x = self.down_sample_block(x)

        for i in range(self.num_encoder_res_blocks):
            stride = 2 if i in [0, 1, 3, 5] else 1
            if i > 0 and i % 3 == 0 and self.use_global_context_mixing:
                x = self.mix_global_context_s1(x) if stride == 1 else self.mix_global_context_s2(x)
                continue

            if self.use_separable_conv_layers:
                x = self.separable_residual_block_s2(x) if stride == 2 else self.separable_residual_block_s1(x)
            else:
                x = self.residual_block_s1(x) if stride == 1 else self.residual_block_s2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_channels: int, nonlinearity: callable, dropout: float = 0):
        super().__init__()

        self.conv_transpose1 = nn.Sequential(
            nn.ConvTranspose2d(num_channels, num_channels // 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_channels // 2),
            nonlinearity,
            nn.Dropout2d(dropout),
        )
        self.conv_transpose2 = nn.Sequential(
            nn.ConvTranspose2d(num_channels // 4, num_channels // 8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_channels // 8),
            nonlinearity,
            nn.Dropout2d(dropout),
        )
        self.decoder = nn.Sequential(
            self.conv_transpose1,
            conv_bn(num_channels // 2, num_channels // 4, kernel_size=3, stride=1, padding=1),
            nonlinearity,
            nn.Dropout2d(dropout),
            self.conv_transpose2,
            conv_bn(num_channels // 8, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)[:, 0, :, :]


class ValueHead(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_conv_bn_blocks: int,
        nonlinearity: callable,
        dropout: float = 0,
        use_reward_target: bool = False,
        use_global_context_mixing: bool = True,
        num_global_pooling_channels: int = 32,
    ):
        super().__init__()

        self.num_conv_bn_blocks = num_conv_bn_blocks
        self.use_reward_target = use_reward_target

        self.dropout = nn.Dropout2d(dropout)
        self.global_pooling_layer = GlobalPooling()
        self.conv_bn_block = nn.Sequential(
            conv_bn(num_channels, num_channels, kernel_size=3, stride=1, padding=1), nonlinearity,
        )
        self.use_global_context_mixing = use_global_context_mixing
        self.mix_global_context = MixGlobalContext(num_channels, nonlinearity, dropout, num_global_pooling_channels)
        self.head = nn.Sequential(nn.Linear(2 * num_channels, 1), nonlinearity, nn.Softplus())

    def forward(self, x):
        for i in range(self.num_conv_bn_blocks):
            if i == 0 and self.use_global_context_mixing:
                x = self.mix_global_context(x)
                continue

            x = self.conv_bn_block(x)
            if self.dropout.p > 0:
                x = self.dropout(x)

        x = self.global_pooling_layer(x)
        x_value = self.head(x)

        x_reward = None
        if self.use_reward_target:
            x_reward = self.head(x)

        return x_value, x_reward


class PolicyHead(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_conv_bn_blocks: int,
        nonlinearity: callable,
        num_actions: int,
        dropout: float = 0,
        mask_policy_head: bool = True,
        use_global_context_mixing: bool = True,
        num_global_pooling_channels: int = 32,
    ):
        super().__init__()

        self.num_conv_bn_blocks = num_conv_bn_blocks
        self.mask_policy_head = mask_policy_head

        self.dropout = nn.Dropout2d(dropout)
        self.global_pooling_layer = GlobalPooling()
        self.conv_bn_block = nn.Sequential(
            conv_bn(num_channels, num_channels, kernel_size=3, stride=1, padding=1), nonlinearity,
        )

        self.use_global_context_mixing = use_global_context_mixing
        self.mix_global_context = MixGlobalContext(num_channels, nonlinearity, dropout, num_global_pooling_channels)
        self.head = nn.Sequential(nn.Linear(2 * num_channels, num_actions))
        self.output_fn = nn.LogSoftmax(dim=1)

    def forward(self, x, valid_actions_msk):
        for i in range(self.num_conv_bn_blocks):
            if i == 0 and self.use_global_context_mixing:
                x = self.mix_global_context(x)
                continue

            x = self.conv_bn_block(x)
            if self.dropout.p > 0:
                x = self.dropout(x)

        x = self.global_pooling_layer(x)
        output = self.head(x)

        if self.mask_policy_head:
            output -= (1 - valid_actions_msk) * 1000

        return self.output_fn(output)
