import torch
from torch import nn


class AudioAttention(nn.Module):
    def __init__(self, num_class, num_channels, num_features):  # 8x5x2014
        self.num_class = num_class
        self.num_channels = num_channels
        self.num_features = num_features
        self.fc1 = nn.Linear(self.num_features * self.num_channels, self.num_features * self.num_channels)
        self.activation = nn.Softmax()

    def forward(self, input):
        x = input.view(self.num_channels * self.num_features)
        x = self.fc1(x)
        x = self.activation(x)  # ho tutti pesi limitati tra 0 e 1
        output = x.view(self.num_channels, self.num_features)
        return output
