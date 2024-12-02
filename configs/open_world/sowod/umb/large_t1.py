_base_ = ('../../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
class_names = [
    # t1
    "airplane","bicycle","bird","boat","bus","car",
    "cat","cow","dog","horse","motorcycle","sheep",
    "train","elephant","bear","zebra","giraffe","truck","person",
    # t2
    "traffic light","fire hydrant","stop sign",
    "parking meter","bench","chair","dining table",
    "potted plant","backpack","umbrella","handbag","tie",
    "suitcase","microwave","oven","toaster","sink","refrigerator","bed","toilet","couch",
    # t3
    "frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake",
    # t4
    "laptop","mouse","remote","keyboard","cell phone",
    "book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush","wine glass","cup","fork","knife","spoon","bowl","tv","bottle",
]
# open world setting
prev_intro_cls = 0
cur_intro_cls = 19
train_json = 'SOWOD/umb_t1_train.json'
setting_path = 'data/VOC2007/SOWOD/t1_setting.pth'
# umb setting
class_distribution_cfg = dict(
    balance=0.1,
    alpha=0.1,
    save_root='UMB/SOWOD/t1_class_distribution',
    known_num=prev_intro_cls+cur_intro_cls,
    att_w=None,
    distribution=None,
)

# yolo world setting
num_classes = 80
num_training_classes = 80
max_epochs = 1  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 1
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4 / 4
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = 'fomo_sowod/t1/epoch_6.pth'
persistent_workers = False

# pipline = [dict(type='att_select', epoch=3, train_par=['att_linear']),
#            dict(type='att_adapt', epoch=3, train_par=['att_embedding']),
#            dict(type='att_refinement', epoch=3, train_par=['att_embedding'])]
pipline = [ dict(type='att_select', epoch=3, train_par=['att_linear']),
            dict(type='att_adapt', epoch=3, train_par=['att_linear', 'att_embedding']),]

# model settings
model = dict(type='UMB',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path='data/texts/mowod_gt.npy',
             prompt_dim=text_channels,
             num_prompts=80,
             setting_path=setting_path,
             known_class_num=prev_intro_cls+cur_intro_cls,
             previsiou_num=prev_intro_cls,
             pipline=pipline,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           frozen_stages=4,
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=True,
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             bbox_head=dict(type='UMBHead',
                            head_module=dict(
                                type='UMBHeadModule',
                                freeze_all=True,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes,
                                ),
                            class_distribution_cfg=class_distribution_cfg),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
coco_train_dataset = dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=class_names),
            data_root='data/VOC2007',
            ann_file=train_json,
            data_prefix=dict(img='JPEGImages/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path=f'data/texts/SOWOD/class_texts.json',
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

custom_hooks = [
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2),
    dict(type='WorkPiplineHook')
]

default_hooks = dict(
    checkpoint=dict(
        interval=save_epoch_intervals, max_keep_ckpts=2, save_best='Current class AP50', rule='greater',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=500,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=1,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'embeddings':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param'))
]
# evaluation settings
test_dataloader = dict(dataset=dict(type='YOLOv5CocoDataset',
                        data_root='data/VOC2007',
                        ann_file='SOWOD/test.json',
                        data_prefix=dict(img='JPEGImages/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=test_pipeline)
                       )

test_evaluator = dict(_delete_=True,
                     type='OWODEvaluator',
                     cfg=dict(
                        dataset_root='data/VOC2007',
                        file_name='sowod/test.txt',
                        prev_intro_cls=prev_intro_cls,
                        cur_intro_cls=cur_intro_cls,
                        unknown_id=80,
                        class_names=class_names
                     )
                    )
val_evaluator = test_evaluator
val_dataloader = test_dataloader
find_unused_parameters = True
