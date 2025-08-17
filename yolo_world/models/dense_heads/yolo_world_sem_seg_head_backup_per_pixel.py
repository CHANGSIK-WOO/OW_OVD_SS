# Copyright (c) Lin Song. All rights reserved.
import math, copy
from typing import List, Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData, PixelData
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptInstanceList, OptMultiConfig, InstanceList
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmyolo.models.dense_heads import YOLOv8HeadModule
from mmyolo.models.utils import gt_instances_preprocess
from mmyolo.registry import MODELS, TASK_UTILS
from mmyolo.models.dense_heads.yolov5_ins_head import ProtoModule, YOLOv5InsHead
from .yolo_world_head import ContrastiveHead, BNContrastiveHead


@MODELS.register_module()
class YOLOWorldSemSegHeadModule(YOLOv8HeadModule):
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

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
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
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.seg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

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
        if self.freeze_bbox or self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        mask_protos = self.proto_pred(img_feats[0]) # (B, proto_channels, H=160, W=160)

        cls_logit, bbox_preds, bbox_dist_preds, coeff_preds = multi_apply(
            self.forward_single, img_feats, txt_feats, self.cls_preds,
            self.reg_preds, self.cls_contrasts, self.seg_preds)
        
        # [25.08.16] sem_seg_logit : semantic segmentation logits using cls_logit's data
        _, _, Hm, Wm = mask_protos.shape
        sem_pyramid = [F.interpolate(x, size=(Hm, Wm), mode='bilinear', align_corners=False) for x in cls_logit]
        sem_seg_logit = torch.stack(sem_pyramid, dim=0).mean(0)  # (B, num_classses, Hm, Wm)

        if self.training:
            #return cls_logit, bbox_preds, bbox_dist_preds, coeff_preds, mask_protos
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_preds, mask_protos, sem_seg_logit
        else:
            #return cls_logit, bbox_preds, None, coeff_preds, mask_protos
            return cls_logit, bbox_preds, None, coeff_preds, mask_protos, sem_seg_logit
        

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,
                       seg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
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

            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
            # (B, h*w, 4) --> (B, 4, h*w) --> (B, 4, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, coeff_pred
        else:
            return cls_logit, bbox_preds, None, coeff_pred

@MODELS.register_module()
class YOLOWorldSemSegHead(YOLOv5InsHead):
    def __init__(self,
                 head_module: ConfigType,
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
                 loss_mask_seg=dict(type='mmdet.CrossEntropyLoss',
                                    use_sigmoid=False,
                                    ignore_index=255,
                                    reduction='mean'),  
                 loss_mask_weight_seg=1.0,                                                   
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
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
        self.loss_mask_seg: nn.Module = MODELS.build(loss_mask_seg)
        self.loss_mask_weight_seg = loss_mask_weight_seg        

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
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
             batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        outs = self(img_feats, txt_feats) # outs = (cls_logit, bbox_preds, bbox_dist_preds, coeff_preds, mask_protos, sem_seg_logit)
        # Fast version
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['masks'],
                              batch_data_samples['img_metas'])
        losses = self.loss_by_feat(*loss_inputs)
        
        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None,
        rescale: bool = False) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        cls_logit, bbox_preds, _, coeff_preds, mask_protos, sem_seg_logit = outs
        predictions = self.predict_by_feat(cls_scores = cls_logit,
                                           bbox_preds = bbox_preds, 
                                           objectnesses = None, 
                                           coeff_preds = coeff_preds, 
                                           proto_preds = mask_protos, 
                                           batch_img_metas = batch_img_metas, 
                                           cfg = proposal_cfg, 
                                           rescale = True, 
                                           with_nms = True)

        # predictions = self.predict_by_feat(*outs,
        #                                    batch_img_metas=batch_img_metas,
        #                                    rescale=rescale)

        B = sem_seg_logit.shape[0] # num_imgs
        for i in range(B):
            meta = batch_img_metas[i]
            ori_h, ori_w = meta['ori_shape'][:2]
            in_h, in_w = meta['batch_input_shape']  # (H_in, W_in)

            # upsampling to input resolutions
            logit_i = sem_seg_logit[i:i+1]  # (1, C, Hm, Wm)
            logit_i = F.interpolate(logit_i, size=(in_h, in_w),
                                    mode='bilinear', align_corners=False)

            # pad crop
            if 'pad_param' in meta and meta['pad_param'] is not None:
                top_pad, bottom_pad, left_pad, right_pad = meta['pad_param']
                top, left = int(top_pad), int(left_pad)
                bottom, right = int(in_h - top_pad), int(in_w - left_pad)
                logit_i = logit_i[:, :, top:bottom, left:right]  # (1, C, H_pad_removed, W_pad_removed)

            # rescaling to original resolutions for demo
            if rescale:
                logit_i = F.interpolate(logit_i, size=(ori_h, ori_w),
                                        mode='bilinear', align_corners=False)

            # per-pixel class prediction using crossentropyloss --> argmax enough 
            pred_sem = logit_i.argmax(dim=1).to(torch.int64).squeeze(0)  # (H, W)

            # append result instance
            predictions[i].pred_sem_seg = PixelData(data=pred_sem)

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
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        #print("[DEBUG] YOLOWorldSegHead_predict")   
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        
        outs = self(img_feats, txt_feats)
        cls_logit, bbox_preds, _, coeff_preds, mask_protos, sem_seg_logit = outs

        #instance results (bbox, mask)
        predictions = self.predict_by_feat(cls_scores = cls_logit,
                                           bbox_preds = bbox_preds, 
                                           objectnesses = None, 
                                           coeff_preds = coeff_preds, 
                                           proto_preds = mask_protos, 
                                           batch_img_metas = batch_img_metas, 
                                           cfg = None, 
                                           rescale = rescale, 
                                           with_nms = True)

        # predictions = self.predict_by_feat(*outs,
        #                                    batch_img_metas=batch_img_metas,
        #                                    rescale=rescale)

        sem_list = list()
        B = sem_seg_logit.shape[0] # num_imgs
        for i in range(B):
            meta = batch_img_metas[i]
            ori_h, ori_w = meta['ori_shape'][:2]
            in_h, in_w = meta['batch_input_shape']  # (H_in, W_in)

            # upsampling to input resolutions
            logit_i = sem_seg_logit[i:i+1]  # (1, C, Hm, Wm)
            logit_i = F.interpolate(logit_i, size=(in_h, in_w), mode='bilinear', align_corners=False)

            # pad crop
            if 'pad_param' in meta and meta['pad_param'] is not None:
                top_pad, bottom_pad, left_pad, right_pad = meta['pad_param']
                top, left = int(top_pad), int(left_pad)
                bottom, right = int(in_h - top_pad), int(in_w - left_pad)
                logit_i = logit_i[:, :, top:bottom, left:right]  # (1, C, H_pad_removed, W_pad_removed)

            # rescaling to original resolutions for demo
            if rescale:
                logit_i = F.interpolate(logit_i, size=(ori_h, ori_w), mode='bilinear', align_corners=False)

            # per-pixel class prediction using crossentropyloss --> argmax enough 
            pred_sem = logit_i.argmax(dim=1).to(torch.int64).squeeze(0)  # (H, W)
            sem_list += [pred_sem]

        return predictions, sem_list

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            coeff_preds: Sequence[Tensor],
            proto_preds: Tensor,
            sem_seg_logit: Tensor, # [25.08.16]
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
        num_imgs = len(batch_img_metas) 

        # ---------- priors ----------
        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

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

        # ---------- GT unpack ----------
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1] # [B, max_inst, 1]
        gt_bboxes = gt_info[:, :, 1:] # [B, max_inst, 4] (xyxy)
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        box_sum_flag = pad_bbox_flag.long().sum(dim=1).squeeze(1)

        # ---------- pred flatten ----------
        flatten_cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,self.num_classes) for cls_pred in cls_scores] 
        flatten_pred_bboxes = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
        flatten_pred_dists = [bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4) for bbox_pred_org in bbox_dist_preds]
        flatten_pred_coeffs = [coeff_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.head_module.mask_channels) for coeff_pred in coeff_preds]


        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1) # flatten_dist_preds : (8, 8400, 64) : 8400 = (80*80) + (40*40) + (20*20) 
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1) # flatten_cls_preds  : (8, 8400, 80)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1) # flatten_pred_bboxes: (8, 8400, 4)
        flatten_pred_bboxes = self.bbox_coder.decode(self.flatten_priors_train[..., :2], flatten_pred_bboxes, self.stride_tensor[..., 0])     
        flatten_pred_coeffs = torch.cat(flatten_pred_coeffs, dim=1) #flatten_pred_coeffs: (8, 8400, 32)

        # ---------- assign ----------
        assigned_result = self.assigner((flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
                                        flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
                                        gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes'] # torch.Size([8, 8400, 4])
        assigned_scores = assigned_result['assigned_scores'] # torch.Size([8, 8400, 80])
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior'] # torch.Size([8, 8400])
        assigned_gt_idxs = assigned_result['assigned_gt_idxs'] # torch.Size([8, 8400])
        
        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        # ---------- cls loss ----------
        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # proto resolution
        _, c, mask_h, mask_w = proto_preds.shape
        if batch_gt_masks.shape[-2:] != (mask_h, mask_w):
            batch_gt_masks = F.interpolate(batch_gt_masks[None], (mask_h, mask_w), mode='nearest')[0]



        assigned_bboxes /= self.stride_tensor #  torch.Size([8, 8400, 4])    
        flatten_pred_bboxes /= self.stride_tensor #  torch.Size([8, 8400, 4])
        num_pos = fg_mask_pre_prior.sum()
   
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error

            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), fg_mask_pre_prior).unsqueeze(-1)

            # bbox loss
            loss_bbox = self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(self.flatten_priors_train[..., :2] / self.stride_tensor, assigned_bboxes, max_dis=self.head_module.reg_max - 1, eps=0.01)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(-1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1, 4).reshape(-1),
                                     avg_factor=assigned_scores_sum)

            #instance mask loss
            loss_mask = torch.zeros(1, device=loss_dfl.device)
            

            batch_inds = torch.zeros(num_imgs,
                                     dtype=torch.int64,
                                     device=assigned_gt_idxs.device)[:, None]
            batch_inds[1:] = box_sum_flag.cumsum(dim=0)[:-1][..., None]
            _assigned_gt_idxs = assigned_gt_idxs + batch_inds


            for bs in range(num_imgs):
                #print("[DEBUG] bs :", bs)                
                # 8400
                bbox_match_inds = assigned_gt_idxs[bs]
                mask_match_inds = _assigned_gt_idxs[bs]
                bbox_match_inds = torch.masked_select(bbox_match_inds, fg_mask_pre_prior[bs])
                mask_match_inds = torch.masked_select(mask_match_inds, fg_mask_pre_prior[bs])
                
                # mask
                mask_dim = coeff_preds[0].shape[1]
                prior_mask_mask = fg_mask_pre_prior[bs].unsqueeze(-1).repeat([1, mask_dim])
                pred_coeffs_pos = torch.masked_select(flatten_pred_coeffs[bs],prior_mask_mask).reshape([-1, mask_dim])
                match_boxes = gt_bboxes[bs][bbox_match_inds] / 4
                normed_boxes = gt_bboxes[bs][bbox_match_inds] / 640
                bbox_area = (normed_boxes[:, 2:] - normed_boxes[:, :2]).prod(dim=1)

                if not mask_match_inds.any():
                    continue
                
                assert not self.mask_overlap
                mask_gti = batch_gt_masks[mask_match_inds]
                mask_preds = (pred_coeffs_pos @ proto_preds[bs].view(c, -1)).view(-1, mask_h, mask_w) # predicted instance masks (proto_pred * seg_pred coeff w.r.t. positive samples)
                loss_mask_full = self.loss_mask(mask_preds, mask_gti) # pixel-wise loss (CrossEntropyLoss) --> (N_pos, mask_h, mask_w)
                _loss_mask = (self.crop_mask(loss_mask_full[None], match_boxes).mean(dim=(2, 3)) / bbox_area)
                loss_mask += _loss_mask.mean()

        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
            loss_mask = flatten_pred_coeffs.sum() * 0


        IGNORE_INDEX = 255
        B = num_imgs
        inst_counts = box_sum_flag

        sem_targets = torch.full(size=(B, mask_h, mask_w), fill_value = IGNORE_INDEX, dtype = torch.int64, device = proto_preds.device)
        offsets = torch.zeros_like(inst_counts) #box_sum_flag : GT Instance count per image
        if num_imgs > 1 :
            offsets[1:] = inst_counts.cumsum(dim=0)[:-1]
        
        gt_cls_per_img = gt_labels.squeeze(-1).long() # [B, max_instance]

        for bs in range(B):
            n = int(inst_counts[bs].item())
            if n == 0 :
                continue
            
            start = int(offsets[bs].item())
            ginds = torch.arange(start, start + n, device=proto_preds.device)
            cls_ids = gt_cls_per_img[bs, :n]

            sem_i = sem_targets[bs]

            for j, gidx in enumerate(ginds):
                m = batch_gt_masks[gidx].bool()
                cid = int(cls_ids[j].item())

                conflict = m & (sem_i != IGNORE_INDEX) & (sem_i != cid)
                sem_i[conflict] = IGNORE_INDEX
                write = m & (sem_i == IGNORE_INDEX)
                sem_i[write] = cid

            sem_targets[bs] = sem_i
        
        if sem_seg_logit.shape[-2:] != (mask_h, mask_w):
            sem_logits = F.interpolate(sem_seg_logit, size=(mask_h, mask_w), mode='bilinear', align_corners=False)

        else:
            sem_logits = sem_seg_logit  # (B, C, Hm, Wm)            

        loss_sem_seg = self.loss_mask_seg(sem_logits, sem_targets)


        _, world_size = get_dist_info()                

        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size,
                    loss_mask=loss_mask * self.loss_mask_weight * world_size,
                    loss_sem_seg=loss_sem_seg * self.loss_mask_weight_seg *world_size,)