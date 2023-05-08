import numpy as np
import torch.nn as nn
import pickle


class TemporalAveragePooling(nn.Module):
    def __init__(self):
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x):
        # calcola la media sui canali (dim=0)
        avg = np.sum(x, 0) / x.shape[0]
        # replica la media per ogni canale
        return avg
