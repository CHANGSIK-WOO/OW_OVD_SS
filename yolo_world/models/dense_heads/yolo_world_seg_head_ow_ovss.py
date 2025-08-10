# Copyright (c) Lin Song. All rights reserved.
import math
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, OptConfigType, OptInstanceList,
                         OptMultiConfig, InstanceList)
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.models.dense_heads import YOLOv8HeadModule
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS, TASK_UTILS
from mmyolo.models.dense_heads.yolov5_ins_head import (
    ProtoModule, YOLOv5InsHead
)

from .yolo_world_head import ContrastiveHead, BNContrastiveHead


@MODELS.register_module()
class OurSegHeadModule(YOLOv8HeadModule):
    def __init__(self,
                 *args,
                 embed_dims: int, 
                 proto_channels: int, # 256 channels
                 mask_channels: int, # 32 channels
                 freeze_bbox: bool = False,
                 freeze_all: bool = False,
                 use_bn_head: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.proto_channels = proto_channels
        self.mask_channels = mask_channels
        self.freeze_bbox = freeze_bbox
        self.freeze_all = freeze_all
        self.use_bn_head = use_bn_head
        super().__init__(*args, **kwargs)
        #print("[DEBUG] YOLOWorldSegHeadModule_Initialize")

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        #print("[DEBUG] YOLOWorldSegHeadModule_initialize weights")
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))      

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        #print("[DEBUG] YOLOWorldSegHeadModule_Initialize Layers")
        # Init decouple head
        self.cls_preds = nn.ModuleList() #class predictions layer [batch, num_classes, H, W]
        self.reg_preds = nn.ModuleList() #Regression(Bounding Box Regression) Predictions layer [batch, 4, H, W]. 4 = [cx, cy, w, h] or [l, t, r, b]
        self.seg_preds = nn.ModuleList() 
        self.cls_contrasts = nn.ModuleList() #Classification Contrastive layer [batch, embed_dim : 256 ~ 1024, H, W]        # NCHW [Number, channel, Height, Width]
        # NCHW [Number, channel, Height, Width]
        # (1) NLP : [batch_size, Seq_len : Token #, Embedding_dim]
        # (2) flatten data : [batch_size, num_anchors, num_channels]
        # (3) Channel size : [batch_size, Sequence_length, channel_size(H*W)] 
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        seg_out_channels = max(self.in_channels[0] // 4, self.mask_channels)
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        bbox_norm_cfg = self.norm_cfg
        bbox_norm_cfg['requires_grad'] = not self.freeze_bbox
        if self.freeze_all:
            self.norm_cfg['requires_grad'] = False
            bbox_norm_cfg['requires_grad'] = False

        for i in range(self.num_levels):
            # (B, 4 * reg_max, H, W)
            # image representation for the bounding box regression
            # reg_max : DFL (Distribution Focal Loss) max value
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            # (B, embed_dims, H, W), embed_dims = text_channels
            # image representation for what class the object belongs to in corresponding grid cell
            self.cls_preds.append( 
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=bbox_norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            # (B, mask_channels, H, W), mask_channels = 32
            self.seg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=seg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=seg_out_channels,
                               out_channels=seg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=seg_out_channels,
                              out_channels=self.mask_channels,
                              kernel_size=1)))

            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims, self.norm_cfg))
            else:
                self.cls_contrasts.append(ContrastiveHead(self.embed_dims)) #initialization of contrastive head
            
        # tensor([0.0, 1.0, 2.0, ..., 15.0]). shape (16, )
        proj = torch.arange(self.reg_max, dtype=torch.float)
        
        self.register_buffer('proj', proj, persistent=False)

        # (B, proto_channels, H, W)
        self.proto_pred = ProtoModule(in_channels=self.in_channels[0],
                                      middle_channels=self.proto_channels,
                                      mask_channels=self.mask_channels,
                                      norm_cfg=self.norm_cfg,
                                      act_cfg=self.act_cfg)
        if self.freeze_bbox or self.freeze_all:
            self._freeze_all()

        


    def _freeze_all(self):
        #print("[DEBUG] YOLOWorldSegHeadModule_freeze all")
        frozen_list = [self.cls_preds, self.reg_preds, self.cls_contrasts]
        if self.freeze_all:
            frozen_list.extend([self.proto_pred, self.seg_preds])
        for module in frozen_list:
            for m in module.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        #print("[DEBUG] YOLOWorldSegHeadModule_train")
        if self.freeze_bbox or self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        #print("[DEBUG] YOLOWorldSegHeadModule_forward")
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        mask_protos = self.proto_pred(img_feats[0])
        cls_logit, bbox_preds, bbox_dist_preds, coeff_preds = multi_apply(
            self.forward_single, img_feats, txt_feats, self.cls_preds,
            self.reg_preds, self.cls_contrasts, self.seg_preds)
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_preds, mask_protos
        else:
            return cls_logit, bbox_preds, None, coeff_preds, mask_protos

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,
                       seg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        #print("[DEBUG] YOLOWorldSegHeadModule_forward_single :")
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat) # image feature map 
        cls_logit = cls_contrast(cls_embed, txt_feat) # cls_embed - txt_feat cosine similarity for cls_logit (class score map)
        bbox_dist_preds = reg_pred(img_feat) 
        coeff_pred = seg_pred(img_feat)
        if self.reg_max > 1: # reg_max = 16 
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            # (B, 4 * reg_max, H, W) -> (B, 4, reg_max, h*w) --> (B, h*w, 4, reg_max=16)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            # (B, h*w, 4, reg_max=16) @ (16, 1) --> (B, h*w, 4, 1) --> (B, h*w, 4)
            
            # self.proj.view([-1, 1]) : (16, 1)
            # proj = tensor([
            #                 [0.0],
            #                 [1.0],
            #                 [2.0],
            #                 [3.0],
            #                 [4.0],
            #                 [5.0],
            #                 [6.0],
            #                 [7.0],
            #                 [8.0],
            #                 [9.0],
            #                 [10.0],
            #                 [11.0],
            #                 [12.0],
            #                 [13.0],
            #                 [14.0],
            #                 [15.0]]) # shape: (16,1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
            # (B, h*w, 4) --> (B, 4, h*w) --> (B, 4, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_pred
        else:
            return cls_logit, bbox_preds, None, coeff_pred


@MODELS.register_module()
class OurSegHead(YOLOv5InsHead):
    def __init__(self,
                 head_module: ConfigType,
                 world_size=-1,
                 att_embeddings=None,
                 prev_intro_cls=0,
                 cur_intro_cls=0,
                 thr=0.8,
                 alpha=0.5,
                 use_sigmoid=True,
                 device="cuda",
                 prev_distribution=None,
                 distributions=None,
                 top_k=10,
                 prior_generator: ConfigType = dict(
                     type='mmdet.MlvlPointGenerator',
                     offset=0.5,
                     strides=[8, 16, 32]),
                 bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                             use_sigmoid=True,
                                             reduction='none',
                                             loss_weight=0.5),
                 loss_bbox: ConfigType = dict(type='IoULoss',
                                              iou_mode='ciou',
                                              bbox_format='xyxy',
                                              reduction='sum',
                                              loss_weight=7.5,
                                              return_iou=False),
                 loss_dfl=dict(type='mmdet.DistributionFocalLoss',
                               reduction='mean',
                               loss_weight=1.5 / 4),
                 mask_overlap: bool = True,
                 loss_mask: ConfigType = dict(type='mmdet.CrossEntropyLoss',
                                              use_sigmoid=True,
                                              reduction='none'),
                 loss_mask_weight=0.05,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        #print("[DEBUG] YOLOWorldSegHead_Initialize")                 
        super().__init__(head_module=head_module,
                         prior_generator=prior_generator,
                         bbox_coder=bbox_coder,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         init_cfg=init_cfg)
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_obj = None
        self.mask_overlap = mask_overlap
        self.loss_mask: nn.Module = MODELS.build(loss_mask)
        self.loss_mask_weight = loss_mask_weight
        self.thr = thr
        self.world_size = world_size
        self.device = device
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid
        self.distributions = distributions
        # self.thrs = [t/100.0 for t in range(50, 100, 5)]
        self.thrs = [thr]
        self.prev_intro_cls = prev_intro_cls
        self.cur_intro_cls = cur_intro_cls
        self.prev_distribution = prev_distribution
        self.top_k = top_k
        self.load_att_embeddings(att_embeddings)        

    def disable_log(self):
        self.positive_distributions = None
        self.negative_distributions = None
        print('disable log')
    
    def enable_log(self):
        self.reset_log()
        print('enable log')
    
    def load_att_embeddings(self, att_embeddings):
        if att_embeddings is None:
            self.att_embeddings = None
            self.disable_log()
            return
        atts = torch.load(att_embeddings)
        self.texts = atts['att_text']
        self.all_atts = atts['att_embedding']
        if self.prev_distribution is not None:
            # todo this
            prev_atts_num = len(torch.load(self.prev_distribution, map_location='cuda')['positive_distributions'][self.thrs.index(self.thr)])
        else:
            prev_atts_num = 0
        self.att_embeddings = torch.nn.Parameter(atts['att_embedding'].float()[prev_atts_num:])
        # self.att_embeddings = torch.nn.Parameter(torch.zeros(1000, 512).float())    
    def reset_log(self, interval=0.0001):
        """Reset the log."""
        # [0, 1] interval = 0.0001
        self.positive_distributions = [{att_i: torch.zeros(int((1)/interval)).to(self.device)
                                    for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        self.negative_distributions=  [{att_i: torch.zeros(int((1)/interval)).to(self.device) 
                                      for att_i in range(self.att_embeddings.shape[0])} for _ in self.thrs]
        
    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        #print("[DEBUG] YOLOWorldSegHead_special_init")   
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            print(f"[DEBUG] YOLOWorldSegHead_Assigner class: {self.assigner.__class__.__name__}")
            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    """YOLO World head."""

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict], fusion_att: bool=False) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        #print("[DEBUG] YOLOWorldSegHead_loss")   
        outs = self(img_feats, txt_feats)
        # do not use att_embeddings
        if self.att_embeddings is None:
            loss_inputs = outs + (None, batch_data_samples['bboxes_labels'],
                                    batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)
            return losses
        
        if fusion_att: 
            num_att = self.att_embeddings.shape[0]
            att_feats = txt_feats[:, -num_att: , :]
            txt_feats = txt_feats[:, :-num_att, :]
        else:
            att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)
        
        
        with torch.no_grad():
            att_outs = self(img_feats, att_feats)[0]
                    
        # Fast version
        loss_inputs = outs + (att_outs, batch_data_samples['bboxes_labels'],
                              batch_data_samples['masks'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)
        
        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        #print("[DEBUG] YOLOWorldSegHead_loss_and_predict")   
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        #print("[DEBUG] YOLOWorldSegHead_forward")   
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False,
                fusion_att: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        #print("[DEBUG] YOLOWorldSegHead_predict")   
        if self.att_embeddings.shape[0] != 25 * (self.num_classes):
            self.select_att()
            
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(img_feats, txt_feats)
        if self.att_embeddings is None:
            predictions = self.predict_by_feat(*outs,
                                               batch_img_metas=batch_img_metas,
                                               rescale=rescale)
            return predictions

        if fusion_att: 
            num_att = self.att_embeddings.shape[0]
            att_feats = txt_feats[:, -num_att: , :]
            txt_feats = txt_feats[:, :-num_att, :]
        else:
            att_feats = self.att_embeddings[None].repeat(txt_feats.shape[0], 1, 1)
        
        if self.att_embeddings is not None:
            outs = self.predict_unknown(outs, img_feats, att_feats)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def fomo_update_outs(self, outs):
        predictions = outs[0]
        ret_logits = []
        for prediction in predictions:
            known_logits = prediction.permute(0, 2, 3, 1)[..., :self.num_classes]
            unknown_logits = prediction.permute(0, 2, 3, 1)[..., :self.num_classes]
            unknown_logits = unknown_logits.max(-1, keepdim=True)[0]
            ret_logits.append(torch.cat([known_logits, unknown_logits], dim=-1).permute(0, 3, 1, 2))
        return (ret_logits, *outs[1:])

    def calculate_uncertainty(self, known_logits):
        known_logits = torch.clamp(known_logits, 1e-6, 1 - 1e-6)
        entropy = (-known_logits * torch.log(known_logits) - (1 - known_logits) * torch.log(1 - known_logits)).mean(dim=-1, keepdim=True)
        return entropy
    
    def select_top_k_attributes(self, adjusted_scores: Tensor, k: int = 3) -> Tensor:
        top_k_scores, _ = adjusted_scores.topk(k, dim=-1)
        top_k_average = top_k_scores.mean(dim=-1, keepdim=True)
        return top_k_average

    def compute_weighted_top_k_attributes(self, adjusted_scores: Tensor, k: int = 10) -> Tensor:
        top_k_scores, top_k_indices = adjusted_scores.topk(k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)
        weighted_average = torch.sum(top_k_scores * top_k_weights, dim=-1, keepdim=True)
        return weighted_average

    def predict_unknown(self, outs, img_feats, att_embeddings):
        known_predictions = outs[0]
        unknown_predictions = self(img_feats, att_embeddings)[0]
        ret_logits = []

        for known_logits, unknown_logits in zip(known_predictions, unknown_predictions):
            known_logits = known_logits.sigmoid().permute(0, 2, 3, 1)
            unknown_logits = unknown_logits.sigmoid().permute(0, 2, 3, 1)

            # 溫←츞藥꿰윥?��?��?���쉪訝띸??�若?����
            uncertainty = self.calculate_uncertainty(known_logits)
            # uncertainty = 0
            # 溫←츞掠?���㏛?���?�若?���㎩뭉瘟껅빐掠욄�㎪?���뇥
            # top_k_att_score = self.select_top_k_attributes(unknown_logits, k=self.top_k)
            top_k_att_score = self.compute_weighted_top_k_attributes(unknown_logits, k=self.top_k)
            #top_k_att_score = unknown_logits.max(dim=-1, keepdim=True)[0]
            # �엻�릦?��꿰윥��?��?���윥?��?��?���쉪�?꾣탩
            
            unknown_logits_final = (top_k_att_score + uncertainty) / 2 * (1 - known_logits.max(-1, keepdim=True)[0])
            # unknown_logits_final = (top_k_att_score) * (1 - known_logits.max(-1, keepdim=True)[0])
            
            # �릦亮뜹?���윥��?��?���윥?��?��?���쉪����???��쥋役?��?���옖
            logits = torch.cat([known_logits, unknown_logits_final], dim=-1).permute(0, 3, 1, 2)
            ret_logits.append(logits)
        
        return (ret_logits, *outs[1:])

    def get_all_dis_sim(self, positive_dis, negative_dis):
        dis_sim = []
        for i in range(len(positive_dis)):
            positive = positive_dis[i]
            negative = negative_dis[i]
            positive = positive / positive.sum()
            negative = negative / negative.sum()
            dis_sim.append(self.get_sim(positive, negative))
        # (num_attributes,)
        return torch.stack(dis_sim).to('cuda')
        
    def combine_distributions(self):
        if self.prev_distribution is None:
            return self.positive_distributions, self.negative_distributions

        # Load previous distributions
        prev_distributions = torch.load(self.prev_distribution, map_location='cuda')
        prev_positive_distributions, prev_negative_distributions = prev_distributions['positive_distributions'], prev_distributions['negative_distributions']

        # Initialize result lists
        ret_pos, ret_neg = prev_positive_distributions, prev_negative_distributions

        # Combine distributions
        for thr in self.thrs:
            thr_id = self.thrs.index(thr)
            if thr_id >= len(prev_positive_distributions) or prev_positive_distributions[thr_id] is None:
                continue
            if thr_id >= len(self.positive_distributions) or self.positive_distributions[thr_id] is None:
                continue
            cur_pos_dist = self.positive_distributions[thr_id]
            cur_neg_dist = self.negative_distributions[thr_id]
            prev_pos_dist = prev_positive_distributions[thr_id]
            prev_neg_dist = prev_negative_distributions[thr_id]
            prev_att = len(prev_pos_dist)
            prev_pos_dist.update({prev_att + k: v for k, v in cur_pos_dist.items()})
            prev_neg_dist.update({prev_att + k: v for k, v in cur_neg_dist.items()})
            ret_pos[thr_id] = prev_pos_dist
            ret_neg[thr_id] = prev_neg_dist
        
        return ret_pos, ret_neg

    def select_att(self, per_class=25):
        """
        Select attributes based on a balance of distribution similarity and attribute diversity.
        Optimized for speed by avoiding redundant calculations and using batch operations.
        """
        
        print(f'thr: {self.thr}')
        # save_root = os.path.dirname(self.distributions)
        # task_id = self.distributions[-5]
        # if not os.path.exists(save_root):
        #     os.makedirs(save_root)
        # torch.save({'positive_distributions': self.positive_distributions,
        #             'negative_distributions': self.negative_distributions}, os.path.join(save_root, f'current{task_id}.pth'))
        # print('save current to {}'.format(os.path.join(save_root, f'current{task_id}.pth')))
        # self.positive_distributions, self.negative_distributions = self.combine_distributions()

        # torch.save({'positive_distributions': self.positive_distributions,
        #             'negative_distributions': self.negative_distributions}, self.distributions)
        # print('save distributions to {}'.format(self.distributions))
        
        distributions = torch.load(self.distributions, map_location='cuda')
        self.positive_distributions, self.negative_distributions = distributions['positive_distributions'], distributions['negative_distributions']
        
        thr_id = self.thrs.index(self.thr)                                                            
        # Step 1: Calculate distribution similarity for each attribute (JS divergence)
        distribution_sim = self.get_all_dis_sim(self.positive_distributions[thr_id], self.negative_distributions[thr_id])
        # Step 2: Prepare for batch cosine similarity calculation
        # Precompute the cosine similarities for all attribute pairs in one batch
        all_atts = self.all_atts.to(self.att_embeddings.device)
        
        att_embeddings_norm = F.normalize(all_atts, p=2, dim=1)  # Normalize embeddings
        if self.use_sigmoid:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).sigmoid() 
        else:
            cosine_sim_matrix = torch.matmul(att_embeddings_norm, att_embeddings_norm.T).abs()
        
        # Initialize selected indices
        selected_indices = []
        
        # Step 3: Attribute selection loop
        for _ in range(per_class * self.num_classes):
            if len(selected_indices) == 0:
                # Select the first attribute with the lowest distribution similarity
                _, idx = distribution_sim.min(dim=0)
            else:
                # Step 4: Calculate diversity score for each unselected attribute
                # Get the mean cosine similarity between unselected and selected attributes
                unselected_indices = list(set(range(len(self.texts))) - set(selected_indices))
                cosine_sim_with_selected = cosine_sim_matrix[unselected_indices][:, selected_indices].mean(dim=1)  # Shape: (num_unselected,)
                
                # Calculate final score: balance distribution similarity and diversity (cosine similarity)
                distribution_sim_unselected = distribution_sim[unselected_indices]
                score = self.alpha * distribution_sim_unselected + (1 - self.alpha) * cosine_sim_with_selected
                
                # Select the attribute with the lowest score
                idx = unselected_indices[score.argmin()]
            
            selected_indices.append(idx)
        
        # Step 5: Update selected attributes and their embeddings
        selected_indices = torch.tensor(selected_indices).to(self.att_embeddings.device)
        self.att_embeddings = torch.nn.Parameter(all_atts[selected_indices]).to(self.att_embeddings.device)
        self.texts = [self.texts[i] for i in selected_indices]
                     
        print('Selected attributes saved.')
  
    def get_sim(self, a, b):
        """
            return distribution a and b similarity. lower value means more similar
        """
        def jensen_shannon_divergence(p, q):
            m = 0.5 * (p + q)
            m = m.clamp(min=1e-6)
            js_div = 0.5 * (torch.sum(p * torch.log((p / m).clamp(min=1e-6))) +
                            torch.sum(q * torch.log((q / m).clamp(min=1e-6))))
            return js_div

        return jensen_shannon_divergence(a, b)
            
    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        #print("[DEBUG] YOLOWorldSegHead_aug_test")   
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            coeff_preds: Sequence[Tensor],
            proto_preds: Tensor,
            att_scores: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_gt_masks: Sequence[Tensor],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        #print("[DEBUG] YOLOWorldSegHead_loss_by_feat")
        num_imgs = len(batch_img_metas) 
        
        #print("[DEBUG] num_imgs :", num_imgs) # num_imgs = 8 = train_batch_size_per_gpu
        #print("[DEBUG] batch_gt_instances :", batch_gt_instances.shape) # torch.Size([145, 6]) : totally 145 instances in 8 images
        #print("[DEBUG] batch_gt_instances :", batch_gt_instances[0]) # tensor([0.0000, 3.0000, 358.9809, 217.5853, 419.2203, 320.5395] : [img_idx, class_id, x1, y1, x2, y2]
        #print("[DEBUG] batch_gt_masks :", batch_gt_masks.shape) # torch.Size([145, 160, 160]) : totally 145 instances' boolean mask (size : 160 * 160) (True : corresponding instances, False : backgrounds)
        #print("[DEBUG] batch_gt_instances :", batch_gt_masks[0]) 
        #print("[DEBUG] batch_img_metas :", batch_img_metas) #[{'batch_input_shape': torch.Size([640, 640])}, {'batch_input_shape': torch.Size([640, 640])}, {'batch_input_shape': torch.Size([640, 640])}, {'batch_input_shape': torch.Size([640, 640])}, {'batch_input_shape': torch.Size([640, 640])}, {'batch_input_shape': torch.Size([640, 640])}, {'batch_input_shape': torch.Size([640, 640])}, {'batch_input_shape': torch.Size([640, 640])}]        
        #print("[DEBUG] batch_gt_instances_ignore :", batch_gt_instances_ignore) #batch_gt_instances_ignore : None   

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        #print("[DEBUG] current_featmap_sizes :", current_featmap_sizes) # current_featmap_sizes : [torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        #print("[DEBUG] gt_info :", gt_info.shape) # torch.Size([8, 42, 5]) : 8images in batch, maximum 42 gt instances per image, 5 info (class_id, x1, y1, x2, y2)
        #print("[DEBUG] gt_info :", gt_info[0])
        gt_labels = gt_info[:, :, :1]
        #print("[DEBUG] gt_labels :", gt_labels.shape) # torch.Size([8, 42, 1])
        #print("[DEBUG] gt_labels :", gt_labels[0]) # class_id 
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        #print("[DEBUG] gt_bboxes :", gt_bboxes.shape) # torch.Size([8, 42, 4])
        #print("[DEBUG] gt_bboxes :", gt_bboxes[0]) 

        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        #print("[DEBUG] pad_bbox_flag :", pad_bbox_flag.shape) # pad_bbox_flag : torch.Size([8, 42, 1])
        #print("[DEBUG] pad_bbox_flag :", pad_bbox_flag[0]) # 1 = GT, 0 = PADDING(NO GT)

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ] 
        #print("[DEBUG] flatten_cls_preds :", flatten_cls_preds[0].shape) #torch.Size([8, 6400, 80]) : 8IMAGES IN BATCH, 6400 GRID CELLS PER IMAGE, LOGITS ABOUT 80CLASSES
        #print("[DEBUG] flatten_cls_preds :", flatten_cls_preds[0])
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        #print("[DEBUG] flatten_pred_bboxes :", flatten_pred_bboxes[0].shape) # torch.Size([8, 6400, 4]) : 8IMAGES IN BATCH, 6400 GRID CELLS PER IMAGE, BBOX PER GRID CELL
        #print("[DEBUG] flatten_pred_bboxes :", flatten_pred_bboxes[0])
        
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]
        #print("[DEBUG] flatten_pred_dists :", flatten_pred_dists[0].shape) #  torch.Size([8, 6400, 64])
        #print("[DEBUG] flatten_pred_dists :", flatten_pred_dists[0])


        flatten_pred_coeffs = [
            coeff_pred.permute(0, 2, 3,
                               1).reshape(num_imgs, -1,
                                          self.head_module.mask_channels)
            for coeff_pred in coeff_preds
        ]
        #print("[DEBUG] flatten_pred_coeffs :", flatten_pred_coeffs[0].shape) # torch.Size([8, 6400, 32]) : 32 MASK COEFFICIENT DIM (MASK CHANNELS)
        #print("[DEBUG] flatten_pred_coeffs :", flatten_pred_coeffs[0])

        if self.att_embeddings is not None:
            # att 
            flatten_att_scores = [att_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,att_score.shape[1])for att_score in att_scores]
            flatten_att_scores = torch.cat(flatten_att_scores, dim=1)
        else:
            flatten_att_scores = None

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1) # flatten_dist_preds : (8, 8400, 64) : 8400 = (80*80) + (40*40) + (20*20) 
        #print("[DEBUG] after cat flatten_dist_preds :", flatten_dist_preds.shape)
        #print("[DEBUG] after cat flatten_dist_preds :", flatten_dist_preds[0])

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1) # flatten_cls_preds  : (8, 8400, 80)
        #print("[DEBUG] after cat flatten_cls_preds :", flatten_cls_preds.shape)
        #print("[DEBUG] after cat flatten_cls_preds :", flatten_cls_preds[0])

        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1) # flatten_pred_bboxes: (8, 8400, 4)
        #print("[DEBUG] after cat flatten_pred_bboxes :", flatten_pred_bboxes.shape)
        #print("[DEBUG] after cat flatten_pred_bboxes :", flatten_pred_bboxes[0])


        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])
        #print("[DEBUG] after decoding flatten_pred_bboxes :", flatten_pred_bboxes.shape)
        #print("[DEBUG] after decoding flatten_pred_bboxes :", flatten_pred_bboxes[0])        

        flatten_pred_coeffs = torch.cat(flatten_pred_coeffs, dim=1) #flatten_pred_coeffs: (8, 8400, 32)
        #print("[DEBUG] after cat flatten_pred_coeffs :", flatten_pred_coeffs.shape)
        #print("[DEBUG] after cat flatten_pred_coeffs :", flatten_pred_coeffs[0])



        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes'] # torch.Size([8, 8400, 4])
        #print("[DEBUG] assigned_bboxes :", assigned_bboxes.shape)
        #print("[DEBUG] assigned_bboxes :", assigned_bboxes[0])


        assigned_scores = assigned_result['assigned_scores'] # torch.Size([8, 8400, 80])
        #print("[DEBUG] assigned_scores :", assigned_scores.shape)
        #print("[DEBUG] assigned_scores :", assigned_scores[0])


        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior'] # torch.Size([8, 8400])
        #print("[DEBUG] fg_mask_pre_prior :", fg_mask_pre_prior.shape)
        #print("[DEBUG] fg_mask_pre_prior :", fg_mask_pre_prior[0])


        assigned_gt_idxs = assigned_result['assigned_gt_idxs'] # torch.Size([8, 8400])
        #print("[DEBUG] assigned_gt_idxs :", assigned_gt_idxs.shape)
        #print("[DEBUG] assigned_gt_idxs :", assigned_gt_idxs[0])


        assigned_scores_sum = assigned_scores.sum().clamp(min=1)
        #print("[DEBUG] assigned_scores_sum :", assigned_scores_sum)        
        self.log_distribution(flatten_att_scores, assigned_scores)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        #print("[DEBUG] loss_cls :", loss_cls)                
        loss_cls /= assigned_scores_sum
        #print("[DEBUG] afterloss_cls :", loss_cls)                


        # rescale bbox
        assigned_bboxes /= self.stride_tensor #  torch.Size([8, 8400, 4])
        #print("[DEBUG] assigned_bboxes :", assigned_bboxes.shape)                
        #print("[DEBUG] assigned_bboxes :", assigned_bboxes[0])        
        flatten_pred_bboxes /= self.stride_tensor #  torch.Size([8, 8400, 4])
        #print("[DEBUG] flatten_pred_bboxes :", flatten_pred_bboxes.shape)                
        #print("[DEBUG] flatten_pred_bboxes :", flatten_pred_bboxes[0])                        

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        #print("[DEBUG] num_pos :", num_pos) # tensor(1057, device='cuda:0')
   
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)

            _, c, mask_h, mask_w = proto_preds.shape
            if batch_gt_masks.shape[-2:] != (mask_h, mask_w):
                batch_gt_masks = F.interpolate(batch_gt_masks[None],
                                               (mask_h, mask_w),
                                               mode='nearest')[0]

            loss_mask = torch.zeros(1, device=loss_dfl.device)
            box_sum_flag = pad_bbox_flag.long().sum(dim=1).squeeze(1)

            batch_inds = torch.zeros(num_imgs,
                                     dtype=torch.int64,
                                     device=assigned_gt_idxs.device)[:, None]
            batch_inds[1:] = box_sum_flag.cumsum(dim=0)[:-1][..., None]
            _assigned_gt_idxs = assigned_gt_idxs + batch_inds


            for bs in range(num_imgs):
                #print("[DEBUG] bs :", bs)                
                # 8400
                bbox_match_inds = assigned_gt_idxs[bs]
                #print("[DEBUG] bbox_match_inds :", bbox_match_inds.shape) 
                #print("[DEBUG] bbox_match_inds :", bbox_match_inds)                
                               
                mask_match_inds = _assigned_gt_idxs[bs]
                #print("[DEBUG] mask_match_inds :", mask_match_inds.shape)                
                #print("[DEBUG] mask_match_inds :", mask_match_inds)                                

                bbox_match_inds = torch.masked_select(bbox_match_inds,
                                                      fg_mask_pre_prior[bs])
                #print("[DEBUG] bbox_match_inds :", bbox_match_inds.shape)
                #print("[DEBUG] bbox_match_inds :", bbox_match_inds)                
                                

                mask_match_inds = torch.masked_select(mask_match_inds,
                                                      fg_mask_pre_prior[bs])
                #print("[DEBUG] mask_match_inds :", mask_match_inds.shape)                
                #print("[DEBUG] mask_match_inds :", mask_match_inds)                                


                # mask
                mask_dim = coeff_preds[0].shape[1]
                #print("[DEBUG] coeff_preds[0] :", coeff_preds[0])                
                #print("[DEBUG] mask_dim :", mask_dim)                
                
                prior_mask_mask = fg_mask_pre_prior[bs].unsqueeze(-1).repeat(
                    [1, mask_dim])
                #print("[DEBUG] fg_mask_pre_prior[bs] :", fg_mask_pre_prior[bs].shape)  
                #print("[DEBUG] fg_mask_pre_prior[bs].unsqueeze(-1) :", fg_mask_pre_prior[bs].unsqueeze(-1).shape)                            
                #print("[DEBUG] fg_mask_pre_prior[bs].unsqueeze(-1).repeat([1, mask_dim]) :", fg_mask_pre_prior[bs].unsqueeze(-1).repeat([1, mask_dim]).shape)                                            
                #print("[DEBUG] prior_mask_mask :", prior_mask_mask)     
                pred_coeffs_pos = torch.masked_select(flatten_pred_coeffs[bs],
                                                      prior_mask_mask).reshape(
                                                          [-1, mask_dim])
                #print("[DEBUG] pred_coeffs_pos :", pred_coeffs_pos.shape)                                                           

                match_boxes = gt_bboxes[bs][bbox_match_inds] / 4
                normed_boxes = gt_bboxes[bs][bbox_match_inds] / 640
                #print("[DEBUG] gt_bboxes[bs] :", gt_bboxes[bs].shape)

                bbox_area = (normed_boxes[:, 2:] -
                             normed_boxes[:, :2]).prod(dim=1)
                if not mask_match_inds.any():
                    continue
                assert not self.mask_overlap
                mask_gti = batch_gt_masks[mask_match_inds]
                #print("[DEBUG] mask_gti :", mask_gti.shape)
                mask_preds = (
                    pred_coeffs_pos @ proto_preds[bs].view(c, -1)).view(
                        -1, mask_h, mask_w) # predicted instance masks (proto_pred * seg_pred coeff w.r.t. positive samples)
                #print("[DEBUG] mask_preds :", mask_preds.shape)                        
                loss_mask_full = self.loss_mask(mask_preds, mask_gti) # pixel-wise loss (CrossEntropyLoss) --> (N_pos, mask_h, mask_w)
                #print("[DEBUG] loss_mask_full :", loss_mask_full.shape)  
                _loss_mask = (self.crop_mask(loss_mask_full[None],
                                             match_boxes).mean(dim=(2, 3)) /
                              bbox_area)
                #print("[DEBUG] before loss_mask :", loss_mask.shape) 
                loss_mask += _loss_mask.mean()                
                #print("[DEBUG] _loss_mask.mean() :", _loss_mask.mean().shape)  
                #print("[DEBUG] after loss_mask :", loss_mask.shape) 

        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
            loss_mask = flatten_pred_coeffs.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        #print("[DEBUG] finish calculating loss by feat") 

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size,
                    loss_mask=loss_mask * self.loss_mask_weight * world_size)
    
    def log_distribution(self, att_scores, assigned_scores):
        
        if not self.training or self.positive_distributions is None \
            or self.att_embeddings is None:
            return
        
        num_att = att_scores.shape[-1]
        num_known = assigned_scores.shape[-1]
        att_scores = att_scores.sigmoid().reshape(-1, num_att).float()      
        assigned_scores = assigned_scores.reshape(-1, num_known)
        # set previous classes to 0
        assigned_scores[:, 0: self.prev_intro_cls] = 0
        assigned_scores = assigned_scores.max(-1)[0]
        for idx, thr in enumerate(self.thrs):
            positive = (assigned_scores >= thr)
            positive_scores = att_scores[positive]
            negative_scores = att_scores[~positive]
            for att_i in range(num_att):
                self.positive_distributions[idx][att_i] += torch.histc(positive_scores[:, att_i], bins=int(1/0.0001), min=0, max=1)
                self.negative_distributions[idx][att_i] += torch.histc(negative_scores[:, att_i], bins=int(1/0.0001), min=0, max=1)    

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        coeff_preds: Optional[List[Tensor]] = None,
                        proto_preds: Optional[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted from the head into
        bbox results.
        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            coeff_preds (list[Tensor]): Mask coefficients predictions
                for all scale levels, each is a 4D-tensor, has shape
                (batch_size, mask_channels, H, W).
            proto_preds (Tensor): Mask prototype features extracted from the
                mask head, has shape (batch_size, mask_channels, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.
        Returns:
            list[:obj:`InstanceData`]: Object detection and instance
            segmentation results of each image after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, h, w).
        """
        assert len(cls_scores) == len(bbox_preds) == len(coeff_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        num_classes = cls_scores[0].size(1)
        # flatten cls_scores, bbox_preds and objectness
        if self.att_embeddings is not None:
            flatten_cls_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    num_classes)
                for cls_score in cls_scores
            ]   
        else:
            flatten_att_scores = [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    num_classes)
                for cls_score in cls_scores
            ]                             

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_coeff_preds = [
            coeff_pred.permute(0, 2, 3,
                               1).reshape(num_imgs, -1,
                                          self.head_module.mask_channels)
            for coeff_pred in coeff_preds
        ]

        if self.att_embeddings is not None:
            flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        else:
            flatten_cls_scores = torch.cat(flatten_att_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        flatten_coeff_preds = torch.cat(flatten_coeff_preds, dim=1)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(len(featmap_sizes))]

        results_list = []
        for (bboxes, scores, objectness, coeffs, mask_proto,
             img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, flatten_coeff_preds,
                              proto_preds, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            batch_input_shape = img_meta['batch_input_shape']
            input_shape_h, input_shape_w = batch_input_shape
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
                input_shape_withoutpad = (input_shape_h - pad_param[0] -
                                          pad_param[1], input_shape_w -
                                          pad_param[2] - pad_param[3])
            else:
                pad_param = None
                input_shape_withoutpad = batch_input_shape
            scale_factor = (input_shape_withoutpad[1] / ori_shape[1],
                            input_shape_withoutpad[0] / ori_shape[0])

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]
                coeffs = coeffs[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]
                # NOTE: Important
                coeffs *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                h, w = ori_shape[:2] if rescale else img_meta['img_shape'][:2]
                empty_results.masks = torch.zeros(
                    size=(0, h, w), dtype=torch.bool, device=bboxes.device)
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0], coeffs=coeffs))
                labels = results['labels']
                coeffs = results['coeffs']
            else:
                scores, labels, keep_idxs, filtered_results = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(coeffs=coeffs))
                coeffs = filtered_results['coeffs']

            results = InstanceData(
                scores=scores,
                labels=labels,
                bboxes=bboxes[keep_idxs],
                coeffs=coeffs)

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)

            if len(results.bboxes):
                masks = self.process_mask(mask_proto, results.coeffs,
                                          results.bboxes,
                                          (input_shape_h, input_shape_w), True)
                if rescale:
                    if pad_param is not None:
                        # bbox minus pad param
                        top_pad, _, left_pad, _ = pad_param
                        results.bboxes -= results.bboxes.new_tensor(
                            [left_pad, top_pad, left_pad, top_pad])
                        # mask crop pad param
                        top, left = int(top_pad), int(left_pad)
                        bottom, right = int(input_shape_h -
                                            top_pad), int(input_shape_w -
                                                          left_pad)
                        masks = masks[:, :, top:bottom, left:right]
                    results.bboxes /= results.bboxes.new_tensor(
                        scale_factor).repeat((1, 2))

                    fast_test = cfg.get('fast_test', False)
                    if fast_test:
                        masks = F.interpolate(
                            masks,
                            size=ori_shape,
                            mode='bilinear',
                            align_corners=False)
                        masks = masks.squeeze(0)
                        masks = masks > cfg.mask_thr_binary
                    else:
                        masks.gt_(cfg.mask_thr_binary)
                        masks = torch.as_tensor(masks, dtype=torch.uint8)
                        masks = masks[0].permute(1, 2,
                                                 0).contiguous().cpu().numpy()
                        masks = mmcv.imresize(masks,
                                              (ori_shape[1], ori_shape[0]))

                        if len(masks.shape) == 2:
                            masks = masks[:, :, None]
                        masks = torch.from_numpy(masks).permute(2, 0, 1)

                results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
                results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

                results.masks = masks.bool()
                results_list.append(results)
            else:
                h, w = ori_shape[:2] if rescale else img_meta['img_shape'][:2]
                results.masks = torch.zeros(
                    size=(0, h, w), dtype=torch.bool, device=bboxes.device)
                results_list.append(results)
        return results_list
                  