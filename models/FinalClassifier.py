import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def classifier(self, x):
        return x

    def forward(self, x):
        return self.classifier(x), {}
