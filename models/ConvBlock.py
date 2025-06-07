import torch.nn as nn


class ConvBlock(nn.Module):
    """
    卷积块，包含两个卷积层、批归一化、激活函数、最大池化和Dropout。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 pool_size, pool_stride, dropout_p, activation):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2  # 保持输出尺寸与输入相同

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation,
            nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            activation,
            nn.MaxPool1d(pool_size, pool_stride),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        return self.block(x)
