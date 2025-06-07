import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ConvBlock import ConvBlock


class DCC(nn.Module):
    """
    主模型类，包含特征提取部分和分类器部分。
    """

    def __init__(self, num_classes):
        super(DCC, self).__init__()

        # 每个卷积块的一些参数设置
        kernel_size = 8
        conv_stride = 1
        pool_size = 8
        pool_stride = 4

        # 卷积块最后一维的长度
        length_after_extraction = 18

        # 特征提取部分，由多个ConvBlock组成
        self.base = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=32, kernel_size=kernel_size,
                      stride=conv_stride, pool_size=pool_size, pool_stride=pool_stride,
                      dropout_p=0.2, activation=nn.ELU(inplace=True)),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=kernel_size,
                      stride=conv_stride, pool_size=pool_size, pool_stride=pool_stride,
                      dropout_p=0.2, activation=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=kernel_size,
                      stride=conv_stride, pool_size=pool_size, pool_stride=pool_stride,
                      dropout_p=0.2, activation=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=kernel_size,
                      stride=conv_stride, pool_size=pool_size, pool_stride=pool_stride,
                      dropout_p=0.2, activation=nn.ReLU(inplace=True)),
            nn.Flatten(),  # Flatten the tensor to a vector
            nn.Linear(256 * length_after_extraction, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
        )
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.7),  # Dropout layer for regularization
            nn.Linear(512, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, input):
        # 如果输入是二维的，增加一个通道维度
        if input.dim() == 2:
            input = input.unsqueeze(1)

        # 特征提取
        feature_map = self.base(input)

        out = self.classifier(feature_map)
        feature = F.normalize(feature_map, dim=1)

        return out, feature
