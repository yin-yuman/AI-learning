import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, config, num_classes=10):

        super(VGG, self).__init__()

        # 构建卷积特征提取部分
        self.features = self._make_layers(config)

        # 找到最后一个卷积层的输出通道数
        last_channels = 0

        for item in config:
            if item != "M":
                last_channels = item

        # 分类部分
        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(
                last_channels,
                num_classes
            )
        )


    def _make_layers(self, config):

        layers = []

        in_channels = 3

        for item in config:

            # M代表最大池化
            if item == "M":

                layers.append(
                    nn.MaxPool2d(
                        kernel_size=2,
                        stride=2
                    )
                )

            # 数字代表卷积层
            else:

                layers.append(
                    nn.Conv2d(
                        in_channels,
                        item,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    )
                )
                layers.append(
                    nn.BatchNorm2d(item)
                )

                layers.append(
                    nn.ReLU(inplace=True)
                )

                in_channels = item

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# VGG-4

VGG_4 = [

    32,
    32,

    "M",

    64,
    64,

    "M"
]


# VGG-6

VGG_6 = [

    32,
    32,

    "M",

    64,
    64,

    "M",

    128,
    128,

    "M"
]


# VGG-8

VGG_8 = [

    32,
    32,

    "M",

    64,
    64,

    "M",

    128,
    128,
    128,
    128,

    "M"
]


# 创建模型

def vgg4():

    return VGG(VGG_4)


def vgg6():

    return VGG(VGG_6)


def vgg8():

    return VGG(VGG_8)