from torch import nn
from torch.nn.init import *
from torch.autograd import Function
from models.TRNmodule import RelationModuleMultiScale


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
    def __init__(self, num_class, frame_aggregation='trn',
                 train_segments=5, val_segments=5,
                 beta=[1, 1, 1], mu=0,
                 dropout_i=0.5, dropout_v=0.5,
                 fc_dim=1024, baseline_type='video',
                 share_params='Y',
                 ens_DA=None):
        # miss something
        super().__init__()
        self.train_segments = train_segments
        self.val_segments = val_segments
        self.fc_dim = fc_dim
        self.beta = beta  # TA3N-opts
        self.mu = mu  # TA3N-opts
        self.feature_dim = 1024
        self.dropout_rate_i = dropout_i
        self.dropout_rate_v = dropout_v
        self.share_params = share_params
        self.frame_aggregation = frame_aggregation
        self.baseline_type = baseline_type
        self.ens_DA = ens_DA
        self.before_softmax = True
        self.softmax = nn.Softmax

        # Prepare DA
        self._prepare_DA(num_class)  # base model = 'resnet101' ??

    def _prepare_DA(self, num_class):  # convert the model to DA framework

        std = 0.001
        feat_shared_dim = min(self.fc_dim,
                              self.feature_dim)  # if self.add_fc > 0 and self.fc_dim > 0 else self.feature_dim
        feat_frame_dim = feat_shared_dim

        self.relu = nn.ReLU(inplace=True)
        self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
        self.dropout_v = nn.Dropout(p=self.dropout_rate_v)

        # ------ FRAME-LEVEL layers (shared layers + source layers + domain layers) ------#

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

            # BN for the above layers.
            # Per ora non l'abbiamo messo

        # ------ AGGEGATE FRAME-BASED features (frame feature --> video feature) ------#
        if self.frame_aggregation == 'trn':  # TRN multiscale
            self.num_bottleneck = 256  # or 512
            self.TRN = RelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
            self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
            self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)

        # ------ VIDEO-LEVEL layers (source layers + domain layers) ------#
        if self.frame_aggregation == 'avgpool':  # avgpool
            feat_aggregated_dim = feat_shared_dim
        if 'trn' in self.frame_aggregation:  # trn
            feat_aggregated_dim = self.num_bottleneck

        feat_video_dim = feat_aggregated_dim

        # 1. source feature layers (video-level)
        # lO USIAMO?
        # self.fc_feature_video_source = nn.Linear(feat_aggregated_dim, feat_video_dim)
        # normal_(self.fc_feature_video_source.weight, 0, std)
        # constant_(self.fc_feature_video_source.bias, 0)

        # 2. domain feature layers (video-level)
        self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
        normal_(self.fc_feature_domain_video.weight, 0, std)
        constant_(self.fc_feature_domain_video.bias, 0)

        # 3. classifiers (video-level)
        self.fc_classifier_video_source = nn.Linear(feat_video_dim, num_class)
        normal_(self.fc_classifier_video_source.weight, 0, std)
        constant_(self.fc_classifier_video_source.bias, 0)

        if self.ens_DA == 'MCD':
            self.fc_classifier_video_source_2 = nn.Linear(feat_video_dim,
                                                          num_class)  # second classifier for self-ensembling
            normal_(self.fc_classifier_video_source_2.weight, 0, std)
            constant_(self.fc_classifier_video_source_2.bias, 0)

        self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
        normal_(self.fc_classifier_domain_video.weight, 0, std)
        constant_(self.fc_classifier_domain_video.bias, 0)

        # domain classifier for TRN-M
        if self.frame_aggregation == 'trn':  # 'trn-m'
            self.relation_domain_classifier_all = nn.ModuleList()
            for i in range(self.train_segments - 1):
                relation_domain_classifier = nn.Sequential(
                    nn.Linear(feat_aggregated_dim, feat_video_dim),
                    nn.ReLU(),
                    nn.Linear(feat_video_dim, 2)
                )
                self.relation_domain_classifier_all += [relation_domain_classifier]

        if self.share_params == 'N':  # capire meglio utilizzo share.params
            self.fc_feature_video_target = nn.Linear(feat_aggregated_dim, feat_video_dim)
            normal_(self.fc_feature_video_target.weight, 0, std)
            constant_(self.fc_feature_video_target.bias, 0)
            self.fc_feature_video_target_2 = nn.Linear(feat_video_dim, feat_video_dim)
            normal_(self.fc_feature_video_target_2.weight, 0, std)
            constant_(self.fc_feature_video_target_2.bias, 0)
            self.fc_classifier_video_target = nn.Linear(feat_video_dim, num_class)
            normal_(self.fc_classifier_video_target.weight, 0, std)
            constant_(self.fc_classifier_video_target.bias, 0)

        # BN for the above layers # skippato again

        # attention mechanism # skip

    # DEF CHE MANCANO

    '''def train               # TODO
        def aggregate_frames    # TODO
        def final_output        # TODO

        def partialBN(self, enable):    # ?

        def get_trans_attn(self, pred_domain):
        def get_general_attn(self, feat):
        def get_attn_feat_frame
        def get_attn_feat_relation             
        # if all attention >> no
    '''

    def final_output(self, pred, pred_video, num_segments):
        if self.baseline_type == 'video':
            base_out = pred_video
        else:
            base_out = pred

        if not self.before_softmax:
            base_out = self.softmax(base_out)

        output = base_out

        if self.baseline_type == 'tsn':
            if self.reshape:
                base_out = base_out.view((-1, num_segments) + base_out.size()[1:])  # e.g. 16 x 3 x 12 (3 segments)

            output = base_out.mean(1)  # e.g. 16 x 12

        return output

    # Gsd
    def domain_classifier_frame(self, feat, beta):  # beta?
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

    def forward(self, input_source, input_target, is_train=True, reverse=False):
        # batch_source, batch_target non sappiamo se servono , ma sono legati a feat_all_source e feat_all_target
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]

        # initailize lists
        feat_all_source = []
        feat_all_target = []
        pred_domain_all_source = []
        pred_domain_all_target = []

        # boh
        num_segments = self.train_segments if is_train else self.val_segments

        # input_data is a list of tensors --> need to do pre-processing
        feat_base_source = input_source.view(-1, input_source.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
        feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048

        # === shared layers ===#
        # need to separate BN for source & target ==> otherwise easy to overfit to source data
        # if self.add_fc < 1:
        #     raise ValueError(Back.RED + 'not enough fc layer')

        # === MLP === with 1 lvl
        feat_fc_source = self.fc_feature_shared_source(feat_base_source)
        feat_fc_target = self.fc_feature_shared_target(
            feat_base_target) if self.share_params == 'N' else self.fc_feature_shared_source(feat_base_target)

        feat_fc_source = self.relu(feat_fc_source)
        feat_fc_target = self.relu(feat_fc_target)

        feat_fc_source = self.dropout_i(feat_fc_source)
        feat_fc_target = self.dropout_i(feat_fc_target)

        # === adversarial branch (frame-level) (GSD) === #
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, self.beta)
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, self.beta)

        pred_domain_all_source.append(
            pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
        pred_domain_all_target.append(
            pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

        # === source layers (frame-level) === # CHIEDERE (CLASSIFICATORE DI AZIONE)
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

            # adversarial branch GRD
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source,
                                                                                   self.beta)
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target,
                                                                                   self.beta)

            # skippata l'attention
            # attn_relation_source = feat_fc_video_relation_source[:, :,
            #                      0]  # assign random tensors to attention values to avoid runtime error
            # attn_relation_target = feat_fc_video_relation_target[:, :,
            #                      0]  # assign random tensors to attention values to avoid runtime error
            # forse inutile

            # sum up relation features (ignore 1-relation)
            feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
            feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)
            # CHIEDERE SE UTILE

        else:
            raise NotImplementedError

        '''if self.baseline_type == 'video':
            feat_all_source.append(feat_fc_video_source.view((batch_source,) + feat_fc_video_source.size()[-1:]))
            feat_all_target.append(feat_fc_video_target.view((batch_target,) + feat_fc_video_target.size()[-1:]))'''

        # === source layers (video-level) ===#
        feat_fc_video_source = self.dropout_v(feat_fc_video_source)
        feat_fc_video_target = self.dropout_v(feat_fc_video_target)

        if reverse:
            feat_fc_video_source = GradReverse.apply(feat_fc_video_source, self.mu)
            feat_fc_video_target = GradReverse.apply(feat_fc_video_target, self.mu)

        pred_fc_video_source = self.fc_classifier_video_source(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video_target(
            feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source(feat_fc_video_target)

        if self.baseline_type == 'video':  # only store the prediction from classifier 1 (for now)
            feat_all_source.append(pred_fc_video_source.view((batch_source,) + pred_fc_video_source.size()[-1:]))
            feat_all_target.append(pred_fc_video_target.view((batch_target,) + pred_fc_video_target.size()[-1:]))

        # === adversarial branch (video-level) (GVD) === #
        pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, self.beta)
        pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, self.beta)

        pred_domain_all_source.append(
            pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
        pred_domain_all_target.append(
            pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

        # video relation-based discriminator
        if self.frame_aggregation == 'trn':
            num_relation = feat_fc_video_relation_source.size()[1]
            pred_domain_all_source.append(pred_fc_domain_video_relation_source.view(
                (batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
            pred_domain_all_target.append(pred_fc_domain_video_relation_target.view(
                (batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))
        else:
            raise NotImplementedError
            '''pred_domain_all_source.append(
                pred_fc_domain_video_source)  # if not trn-m, add dummy tensors for relation features
            pred_domain_all_target.append(pred_fc_domain_video_target)'''

        # === final output ===#
        output_source = self.final_output(pred_fc_source, pred_fc_video_source,
                                          num_segments)  # select output from frame or video prediction
        output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)

        output_source_2 = output_source
        output_target_2 = output_target

        if self.ens_DA == 'MCD':
            pred_fc_video_source_2 = self.fc_classifier_video_source_2(feat_fc_video_source)
            pred_fc_video_target_2 = self.fc_classifier_video_target_2(
                feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source_2(
                feat_fc_video_target)
            output_source_2 = self.final_output(pred_fc_source, pred_fc_video_source_2, num_segments)
            output_target_2 = self.final_output(pred_fc_target, pred_fc_video_target_2, num_segments)

        # return output_source, output_source_2, pred_domain_all_source[::-1], feat_all_source[::-1], \
        #        output_target, output_target_2, pred_domain_all_target[ ::-1], feat_all_target[::-1]
        # Dani: provo a cambiare output per adattarlo a quello richiesto da action recognition
        # return [pred_domain_all_source[::-1], pred_domain_all_target[::-1]], [feat_all_source[::-1], feat_all_target[::-1]]
        return output_source, {'source': pred_domain_all_source[::-1], 'target': pred_domain_all_target[::-1]}
        # reverse the order of feature list due to some multi-gpu issues
        # attn_relation_source, attn_relation_target,

        # return self.classifier(x), {}  #default
