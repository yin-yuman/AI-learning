import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            # 第一层卷积
            nn.Conv2d(
                in_channels=3,#输入通道RGB
                out_channels=16,#输出通道，学习16种不同特征
                kernel_size=3,#卷积核尺寸
                padding=1
            ),

            nn.ReLU(),##ReLU(x)=max(0,x)

            nn.MaxPool2d(2),#最大池化，缩小图片尺寸

            # 第二层卷积
            nn.Conv2d(
                16,
                32,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )


        self.fc = nn.Sequential(

            nn.Linear(
                32*8*8,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10 #10个类别
            )
        )


    def forward(self,x):

        x = self.conv(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.fc(x)

        return x