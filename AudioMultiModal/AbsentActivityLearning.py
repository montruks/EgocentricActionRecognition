import torch
from torch import nn


def lossAbsentActivity(pred_audio_target): # loss da aggiungere per absent activity
    p = 1 - pred_audio_target
    loss = -torch.log(p).sum(dim=1)
    return loss


class AbsentActivityLearning(nn.Module):
    def __init__(self, num_class, num_channels, num_features, *args, **kwargs):  # 8x5x2014
        super(AbsentActivityLearning).__init__()
        self.num_class = num_class
        self.num_channels = num_channels
        self.num_features = num_features
