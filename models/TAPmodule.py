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


def temporalAveragePooling(path):
    f = open(path, 'rb')
    feature_pkl = pickle.load(f)
    temp_feature_pkl = copy.deepcopy(feature_pkl)
    for k in range(0, len(feature_pkl['features'])):
        feature = feature_pkl['features'][k]['features_RGB']
        # Temporal Pooling Average
        temp_feature_pkl['features'][k]['features_RGB'] = np.sum(feature, axis=0)
    return temp_feature_pkl