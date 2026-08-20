import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        # Residual connection
        out = out + identity

        out = self.relu(out)

        return out


class ResNetSmall(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(
            inplace=True
        )

        # 将通道数从 32 调整到 64
        self.channel_conv = nn.Conv2d(
            32,
            64,
            kernel_size=1,
            bias=False
        )

        self.channel_bn = nn.BatchNorm2d(64)

        # 4 个 Residual Blocks
        self.layers = nn.Sequential(

            ResidualBlock(64),

            ResidualBlock(64),

            ResidualBlock(64),

            ResidualBlock(64)
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.channel_conv(x)

        x = self.channel_bn(x)

        x = self.relu(x)

        x = self.layers(x)

        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc(x)

        return x