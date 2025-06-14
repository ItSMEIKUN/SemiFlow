import torch.nn as nn
import torch.nn.functional as F

from models.ConvBlock import ConvBlock


class DCC(nn.Module):

    """
    Master model class with feature extraction part and classifier part.
    """
    def __init__(self, num_classes):
        super(DCC, self).__init__()

        # Some parameter settings for each convolution block.
        kernel_size = 8
        conv_stride = 1
        pool_size = 8
        pool_stride = 4

        # Length of the last dimension of the convolution block
        length_after_extraction = 18

        # Feature extraction section, consisting of multiple ConvBlocks
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
        # Classifier section
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.7),  # Dropout layer for regularization
            nn.Linear(512, 512, bias=False),  # Fully connected layer
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, input):
        # If the input is two-dimensional, add a channel dimension
        if input.dim() == 2:
            input = input.unsqueeze(1)

        # Feature extraction
        feature_map = self.base(input)

        out = self.classifier(feature_map)
        feature = F.normalize(feature_map, dim=1)

        return out, feature
