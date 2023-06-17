from torch import nn
from torch.nn.init import *
from torch.autograd import Function
from models.FeatureExtractor import FeatureExtractor


# definition of Gradient Reversal Layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class ClassifierMidFusion(nn.Module):
    def __init__(self, num_class, num_modalities=3, num_segments=5, feature_dim=1024, share_params='Y', add_fc=1,
                 frame_aggregation='none', use_attn=False, beta=[1, 1, 1], dropout_i=0.5, dropout_v=0.5,
                 ens_DA=None):

        super().__init__()
        self.num_modalities = num_modalities
        self.share_params = share_params
        self.beta = beta
        self.dropout_rate_v = dropout_v
        self.before_softmax = True

        self.featuresExtractors = []
        for m in range(self.num_modalities):
            self.featuresExtractors.append(FeatureExtractor(num_segments=num_segments, feature_dim=feature_dim,
                                                            share_params=share_params, add_fc=add_fc,
                                                            frame_aggregation=frame_aggregation, use_attn=use_attn,
                                                            beta=beta, dropout_i=dropout_i, ens_DA=ens_DA))

        std = 0.001

        self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

        self.num_bottleneck = 256 * self.num_modalities
        feat_aggregated_dim = feature_dim * self.num_modalities
        if frame_aggregation == 'trn':
            feat_aggregated_dim = self.num_bottleneck

        # 1. fc mid (video-level)
        self.fc_mid_shared_source = nn.Linear(feat_aggregated_dim, feat_aggregated_dim)
        normal_(self.fc_mid_shared_source.weight, 0, std)
        constant_(self.fc_mid_shared_source.bias, 0)

        if self.share_params == 'N':
            self.fc_mid_shared_target = nn.Linear(feat_aggregated_dim, feat_aggregated_dim)
            normal_(self.fc_mid_shared_target.weight, 0, std)
            constant_(self.fc_mid_shared_target.bias, 0)

        # 2. domain feature layers (video-level)
        self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_aggregated_dim)
        normal_(self.fc_feature_domain_video.weight, 0, std)
        constant_(self.fc_feature_domain_video.bias, 0)

        self.fc_classifier_domain_video = nn.Linear(feat_aggregated_dim, 2)
        normal_(self.fc_classifier_domain_video.weight, 0, std)
        constant_(self.fc_classifier_domain_video.bias, 0)

        # 3. classifier (video-level)
        self.fc_classifier_video_source = nn.Linear(feat_aggregated_dim, num_class)
        normal_(self.fc_classifier_video_source.weight, 0, std)
        constant_(self.fc_classifier_video_source.bias, 0)

        if self.share_params == 'N':
            self.fc_classifier_video_target = nn.Linear(feat_aggregated_dim, num_class)
            normal_(self.fc_classifier_video_target.weight, 0, std)
            constant_(self.fc_classifier_video_target.bias, 0)

    # Gvd
    def domain_classifier_video(self, feat_video, beta):
        feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
        feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
        feat_fc_domain_video = self.relu(feat_fc_domain_video)
        pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

        return pred_fc_domain_video

    def forward(self, input_source, input_target):

        # initailize dicts
        pred_domain_all_source = {}
        pred_domain_all_target = {}

        feat_fc_video_source = None
        feat_fc_video_target = None

        for m in range(self.num_modalities):
            pred_domain_source, pred_domain_target, feat_source, feat_target = self.featuresExtractors[m](torch.squeeze(input_source[:, m, :, :]), torch.squeeze(input_target[:, m, :, :]))
            if m == 0:
                feat_fc_video_source = feat_source
                feat_fc_video_target = feat_target
                for k in pred_domain_source.keys():
                    pred_domain_all_source[k] = pred_domain_source[k]
                for k in pred_domain_target.keys():
                    pred_domain_all_target[k] = pred_domain_target[k]
            else:
                feat_fc_video_source = torch.cat((feat_fc_video_source, feat_source), dim=1)
                feat_fc_video_target = torch.cat((feat_fc_video_target, feat_target), dim=1)
                for k in pred_domain_source.keys():
                    pred_domain_all_source[k] = torch.cat((pred_domain_all_source[k], pred_domain_source[k]), dim=1)
                for k in pred_domain_target.keys():
                    pred_domain_all_target[k] = torch.cat((pred_domain_all_target[k], pred_domain_target[k]), dim=1)

        # batch_source, batch_target
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]

        feat_fc_video_source = self.dropout_v(feat_fc_video_source)
        feat_fc_video_target = self.dropout_v(feat_fc_video_target)

        pred_fc_video_source = self.fc_classifier_video_source(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video_target(
            feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source(feat_fc_video_target)

        # === adversarial branch (video-level) (GVD) === #
        pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, self.beta)
        pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, self.beta)

        pred_domain_all_source['GVD'] = pred_fc_domain_video_source.view(
            (batch_source,) + pred_fc_domain_video_source.size()[-1:])
        pred_domain_all_target['GVD'] = pred_fc_domain_video_target.view(
            (batch_target,) + pred_fc_domain_video_target.size()[-1:])

        # === final output ===#
        if not self.before_softmax:
            pred_fc_video_source = self.softmax(pred_fc_video_source)
            pred_fc_video_target = self.softmax(pred_fc_video_target)

        return pred_fc_video_source, pred_domain_all_source, pred_fc_video_target, pred_domain_all_target
