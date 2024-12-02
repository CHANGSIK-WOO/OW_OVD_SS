# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
import torch.nn.functional as F
import json


@MODELS.register_module()
class FOMOnoBN(YOLODetector):
    """Implementation of YOLO World Series
    
        setting_path: str, path to the setting file
        {
            'Linear_weight': torch.nn.Linear(len(self.attributes_texts), self.known_class_num)
            'att_text': [object which ...]
            'att_embedding': tensor
        }
    """

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 setting_path='',
                 device='cuda',
                 known_class_num=80,
                 previsiou_num=0,
                 att_per_class=25,
                 pipline=[dict(type='att_select', epoch=1)],
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        self.setting_path = setting_path
        self.device = device
        self.known_class_num = known_class_num
        self.previsiou_num = previsiou_num
        self.pipline = pipline 
        self.att_per_class = att_per_class
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

        
        self.get_default_setting()

    def get_default_setting(self):
        setting = torch.load(self.setting_path)
        self.att_texts = setting['att_text']
        self.att_embedding = torch.nn.Parameter(setting['att_embedding'].float())
        self.vaild_att = torch.nn.Parameter(torch.ones(len(self.att_texts), dtype=torch.float))
        self.bbox_head.set_linear(setting['Linear_weight'])
        self.vaild_att.requires_grad = False
        
        vaild = torch.nn.Parameter(torch.zeros(self.num_training_classes, dtype=torch.float))
        vaild.requires_grad = False
        vaild[self.previsiou_num: self.known_class_num] = 1
        self.bbox_head.set_vaild(vaild)
        if self.pipline[0]['type'] == 'att_select':
            self.bbox_head.set_collect(self.pipline[0]['mean_embedding'])

    def fomo_train(self, batch_inputs: Tensor,
             batch_data_samples: SampleList):
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats,
                                            batch_data_samples)
        return losses
      
    def prepare_adapt(self):
        collect_embeddings = self.bbox_head.collect_embedding
        mean_embeddings = []
        for i in range(self.previsiou_num, self.known_class_num):
            mean_embeddings.append(collect_embeddings[i][0] / collect_embeddings[i][1])
        # (num_known, 512)
        self.mean_embeddings = torch.stack(mean_embeddings)
        self.mean_embeddings.requires_grad = False
                  
    def adapt_train(self, batch_inputs: Tensor,
                batch_data_samples: SampleList):
        # num_cur, num_att
        w = self.bbox_head.att_linear.weight[self.previsiou_num:self.known_class_num, :]
        output = (w @ self.att_embedding)
        loss = (self.mean_embeddings - output).pow(2).mean()
        return {'adapt_loss': loss}
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        if self.pipline[0]['type'] != 'att_adapt':
            return self.fomo_train(batch_inputs, batch_data_samples)
        else:
            return self.adapt_train(batch_inputs, batch_data_samples)
        
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes

        results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        att_embeddings = self.att_embedding[None]
        att_embeddings = att_embeddings.repeat(img_feats[0].shape[0], 1, 1)
        
        img_feats = self.neck(img_feats, att_embeddings)

        return img_feats, att_embeddings
    
    def select_att(self):
        """ select att
        """
        # (num_att, num_known)
        att_weights = self.bbox_head.att_linear.weight.T
        ori_att_weights = att_weights.clone()
        num_att, num_known = att_weights.shape
        with torch.no_grad():
            att_weights[...,  :self.previsiou_num] = 0
            att_weights[..., self.known_class_num:] = 0
            
            att_weights = att_weights.reshape(-1)
            num_vaild = self.known_class_num - self.previsiou_num
            _, top_indices = torch.topk(att_weights, num_vaild * self.att_per_class)
            att_weights.fill_(0)  # Reset all attributes to 0
            att_weights[top_indices] = 1
            att_weights = att_weights.reshape(num_att, num_known)
            
            selected_idx = torch.where(torch.sum(att_weights, dim=1) != 0)[0]
            self.att_embedding = torch.nn.Parameter(torch.index_select(self.att_embedding, 0, selected_idx))
            ori_att_weights = torch.index_select(ori_att_weights, 0, selected_idx)
            ori_att_weights = F.normalize(ori_att_weights, p=1, dim=0).to(ori_att_weights)
            self.bbox_head.set_linear({'weight': ori_att_weights.T})
            print(f"Selected {len(selected_idx.tolist())} attributes from {len(self.att_texts)}")
            self.att_texts = [self.att_texts[i] for i in selected_idx.tolist()]
            self.vaild_att = torch.nn.Parameter(torch.zeros_like(self.vaild_att))
            self.vaild_att[selected_idx] = 0
    
    def set_train_linear(self, enable):
        for par in self.bbox_head.att_linear.parameters():
            par.requires_grad = enable
            
    def set_train_embedding(self, enable):
        self.att_embedding.requires_grad = enable

    def get_obj(self):
        # (num_known, num_att)
        att_weights = self.bbox_head.att_linear.weight[self.previsiou_num:self.known_class_num, :]
        # (num_att, 512)
        att_embedding = self.att_embedding
        obj = (att_weights @ att_embedding).mean(dim=0)                   
        # (num_att+1, 512) 
        self.att_embedding = torch.nn.Parameter(torch.cat([self.att_embedding, obj[None]], dim=0))
        # (num_att+1, known_class+1) 
        eye_unknown = torch.eye(1, device=self.device)
        att_weights = torch.block_diag(self.bbox_head.att_linear.weight, eye_unknown)
        self.bbox_head.set_linear({'weight': att_weights})
        self.bbox_head.valid = torch.nn.Parameter(torch.cat([self.bbox_head.valid, 
                                                             torch.tensor([1.0], device=self.bbox_head.valid.device)]))
