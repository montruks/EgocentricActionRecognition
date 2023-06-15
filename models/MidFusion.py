from torch import nn
from torch.nn.init import *
from torch.autograd import Function


# Sommo le varie feature e calcolo classificatori
class MidFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_modalities):
        modalities = features_modalities.keys()
        n_mod = len(modalities)  # numero di modalità

        # Estraggo Features
        feat_fc_target_list = []
        feat_fc_source_list = []
        feat_fc_video_source_list = []
        feat_fc_video_target_list = []
        for m in modalities:
            feat_fc_source_list.append(features_modalities[m]['source']['FC'])
            feat_fc_target_list.append(features_modalities[m]['target']['FC'])
            feat_fc_video_source_list.append(features_modalities[m]['source']['Aggregation'])
            feat_fc_video_target_list.append(features_modalities[m]['target']['Aggregation'])

        # Sommo rispetto alle modalità
        feat_fc_source_TBN = torch.zeros_like(feat_fc_source_list[0])
        feat_fc_target_TBN = torch.zeros_like(feat_fc_target_list[0])
        feat_fc_source_video_TBN = torch.zeros_like(feat_fc_video_source_list[0])
        feat_fc_target_video_TBN = torch.zeros_like(feat_fc_video_target_list[0])
        for i in range(n_mod):
            feat_fc_source_TBN += feat_fc_source_list[i]
            feat_fc_target_TBN += feat_fc_target_list[i]
            feat_fc_source_video_TBN += feat_fc_video_source_list[i]
            feat_fc_target_video_TBN += feat_fc_video_target_list[i]

        # Ne faccio la media
        feat_fc_source_TBN = feat_fc_source_TBN/n_mod
        feat_fc_target_TBN = feat_fc_target_TBN/n_mod
        feat_fc_source_video_TBN = feat_fc_source_video_TBN/n_mod
        feat_fc_target_video_TBN = feat_fc_target_video_TBN/n_mod

        return feat_fc_source_TBN, feat_fc_target_TBN, feat_fc_source_video_TBN, feat_fc_target_video_TBN



