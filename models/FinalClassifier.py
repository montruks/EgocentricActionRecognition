from torch import nn
from torch.nn.init import *


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



class Classifier(nn.Module):
    def __init__(self, num_classes,train_segments=5, val_segments=5, beta=[1,1,1], frame_aggregation = 'trn', dropout_i=0.5, dropout_v=0.5, fc_dim=1024 ,share_params='Y'):
        super().__init__()
        self.train_segments = train_segments
        self.val_segments = val_segments
        self.fc_dim = fc_dim
        self.beta = beta  # TA3N-opts
        self.feature_dim = 1024
        self.dropout_rate_i = dropout_i
        self.dropout_rate_v = dropout_v
        self.share_params = share_params
        self.frame_aggregation = frame_aggregation

    def _prepare_DA(self, num_class, base_model):  # convert the model to DA framework

        std = 0.001
        feat_shared_dim = min(self.fc_dim, self.feature_dim)# if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
        feat_frame_dim = feat_shared_dim

        self.relu = nn.ReLU(inplace=True)
        self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
        self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

        # 1. shared feature layers
        self.fc_feature_shared_source = nn.Linear(self.feature_dim, feat_shared_dim)
        normal_(self.fc_feature_shared_source.weight, 0, std)
        constant_(self.fc_feature_shared_source.bias, 0)

        # 2. frame-level feature layers
        self.fc_feature_source = nn.Linear(feat_shared_dim, feat_frame_dim)
        normal_(self.fc_feature_source.weight, 0, std)
        constant_(self.fc_feature_source.bias, 0)

        # 3. domain feature layers (frame-level)
        self.fc_feature_domain = nn.Linear(feat_shared_dim, feat_frame_dim)
        normal_(self.fc_feature_domain.weight, 0, std)
        constant_(self.fc_feature_domain.bias, 0)

        # 4. classifiers (frame-level)
        self.fc_classifier_source = nn.Linear(feat_frame_dim, num_class)
        normal_(self.fc_classifier_source.weight, 0, std)
        constant_(self.fc_classifier_source.bias, 0)

        self.fc_classifier_domain = nn.Linear(feat_frame_dim, 2)
        normal_(self.fc_classifier_domain.weight, 0, std)
        constant_(self.fc_classifier_domain.bias, 0)


        if self.share_params == 'N':
            self.fc_feature_shared_target = nn.Linear(self.feature_dim, feat_shared_dim)
            normal_(self.fc_feature_shared_target.weight, 0, std)
            constant_(self.fc_feature_shared_target.bias, 0)

            self.fc_feature_target = nn.Linear(feat_shared_dim, feat_frame_dim)
            normal_(self.fc_feature_target.weight, 0, std)
            constant_(self.fc_feature_target.bias, 0)

            self.fc_classifier_target = nn.Linear(feat_frame_dim, num_class)
            normal_(self.fc_classifier_target.weight, 0, std)
            constant_(self.fc_classifier_target.bias, 0)


        # Video Level

        # 2. domain feature layers (video-level)
        self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
        normal_(self.fc_feature_domain_video.weight, 0, std)
        constant_(self.fc_feature_domain_video.bias, 0)

        self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
        normal_(self.fc_classifier_domain_video.weight, 0, std)
        constant_(self.fc_classifier_domain_video.bias, 0)

        # domain classifier for TRN-M
        if self.frame_aggregation == 'trn':   # 'trn-m'
            self.relation_domain_classifier_all = nn.ModuleList()
            for i in range(self.train_segments - 1):
                relation_domain_classifier = nn.Sequential(
                    nn.Linear(feat_aggregated_dim, feat_video_dim),
                    nn.ReLU(),
                    nn.Linear(feat_video_dim, 2)
                )
                self.relation_domain_classifier_all += [relation_domain_classifier]




    # Gsd
    def domain_classifier_frame(self, feat, beta): #beta?
        feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
        feat_fc_domain_frame = self.fc_feature_domain(feat_fc_domain_frame)
        feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
        pred_fc_domain_frame = self.fc_classifier_domain(feat_fc_domain_frame)

        return pred_fc_domain_frame

    # Gvd
    def domain_classifier_video(self, feat_video, beta):
        feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
        feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
        feat_fc_domain_video = self.relu(feat_fc_domain_video)
        pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

        return pred_fc_domain_video

    # Grd
    def domain_classifier_relation(self, feat_relation, beta):
        # 128x4x256 --> (128x4)x2
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:, i, :].squeeze(1)  # 128x1x256 --> 128x256
            feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single,
                                                               beta[0])  # the same beta for all relations (for now)

            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)

            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1, 1, 2)
            else:
                pred_fc_domain_relation_video = torch.cat(
                    (pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1, 1, 2)), 1)

        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1, 2)

        return pred_fc_domain_relation_video

    def forward(self, x):

        # input_data is a list of tensors --> need to do pre-processing
        feat_base_source = input_source.view(-1, input_source.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
        feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048

        # === shared layers ===#
         # need to separate BN for source & target ==> otherwise easy to overfit to source data
          # if self.add_fc < 1:
          #     raise ValueError(Back.RED + 'not enough fc layer')

        feat_fc_source = self.fc_feature_shared_source(feat_base_source)
        feat_fc_target = self.fc_feature_shared_target(
            feat_base_target) if self.share_params == 'N' else self.fc_feature_shared_source(feat_base_target)

        #
        feat_fc_source = self.relu(feat_fc_source)
        feat_fc_target = self.relu(feat_fc_target)
        feat_fc_source = self.dropout_i(feat_fc_source)
        feat_fc_target = self.dropout_i(feat_fc_target)

        # === adversarial branch (frame-level) ===#
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta)
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta)

        # === source layers (frame-level) ===#
        pred_fc_source = self.fc_classifier_source(feat_fc_source)
        pred_fc_target = self.fc_classifier_target(
            feat_fc_target) if self.share_params == 'N' else self.fc_classifier_source(feat_fc_target)

        ### aggregate the frame-based features to video-based features ###
        ''' if self.frame_aggregation == 'avgpool':
            feat_fc_video_source = self.aggregate_frames(feat_fc_source, num_segments, pred_fc_domain_frame_source)
            feat_fc_video_target = self.aggregate_frames(feat_fc_target, num_segments, pred_fc_domain_frame_target)

            attn_relation_source = feat_fc_video_source[:,
                                   0]  # assign random tensors to attention values to avoid runtime error
            attn_relation_target = feat_fc_video_target[:,
                                   0]  # assign random tensors to attention values to avoid runtime error'''

        if 'trn' in self.frame_aggregation:
            feat_fc_video_source = feat_fc_source.view((-1, num_segments) + feat_fc_source.size()[-1:])
                # reshape based on the segments (e.g. 640x512 --> 128x5x512)
            feat_fc_video_target = feat_fc_target.view((-1, num_segments) + feat_fc_target.size()[-1:])
                # reshape based on the segments (e.g. 640x512 --> 128x5x512)

            feat_fc_video_relation_source = self.TRN(
                feat_fc_video_source)  # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
            feat_fc_video_relation_target = self.TRN(feat_fc_video_target)

            # adversarial branch
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)

            # skippata l'attention
            # attn_relation_source = feat_fc_video_relation_source[:, :,
            #                      0]  # assign random tensors to attention values to avoid runtime error
            # attn_relation_target = feat_fc_video_relation_target[:, :,
            #                      0]  # assign random tensors to attention values to avoid runtime error
            # forse inutile

            # sum up relation features (ignore 1-relation)
            feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
            feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)

        else:
            raise NotImplementedError
            # CHIEDERE SE UTILE

        '''if self.baseline_type == 'video':
            feat_all_source.append(feat_fc_video_source.view((batch_source,) + feat_fc_video_source.size()[-1:]))
            feat_all_target.append(feat_fc_video_target.view((batch_target,) + feat_fc_video_target.size()[-1:]))'''


        # === source layers (video-level) ===#
        feat_fc_video_source = self.dropout_v(feat_fc_video_source)
        feat_fc_video_target = self.dropout_v(feat_fc_video_target)

        if reverse:
            feat_fc_video_source = GradReverse.apply(feat_fc_video_source, mu)
            feat_fc_video_target = GradReverse.apply(feat_fc_video_target, mu)

        pred_fc_video_source = self.fc_classifier_video_source(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video_target(
            feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source(feat_fc_video_target)

        return self.classifier(x), {}
