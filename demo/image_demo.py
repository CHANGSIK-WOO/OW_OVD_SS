# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2 #openCV : read, write, show image
import argparse
import os.path as osp

import torch
import numpy as np
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1) #box instance
MASK_ANNOTATOR = sv.MaskAnnotator() #mask instance


class LabelAnnotator(sv.LabelAnnotator): #label custom annotator

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h
        # (x_min, y_min, x_max, y_max) means the top-left and bottom-right corners of the text box


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


# ======= [ADDED] semantic visualization helpers =======
def _colorize_label_map(label_hw: np.ndarray) -> np.ndarray:
    h, w = label_hw.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in np.unique(label_hw):
        if cls_id < 0:
            continue
        rng = np.random.default_rng(int(cls_id))
        color[label_hw == cls_id] = rng.integers(0, 256, size=3, dtype=np.uint8)
    return color

def save_semantic_overlay(img_bgr: np.ndarray, sem_map, out_path: str, alpha: float = 0.5):
    """
    sem_map: torch.Tensor | np.ndarray
      - (1,H,W) class id map
      - (C,H,W) logit
      - (H,W)   class id map
    """
    if isinstance(sem_map, torch.Tensor):
        sem_map = sem_map.detach().cpu().numpy()
    if sem_map.ndim == 3 and sem_map.shape[0] > 1:
        # (C,H,W) → argmax
        sem_map = sem_map.argmax(0)
    elif sem_map.ndim == 3 and sem_map.shape[0] == 1:
        # (1,H,W) → (H,W)
        sem_map = sem_map[0]
    sem_map = sem_map.astype(np.int32)

    H, W = img_bgr.shape[:2]
    if sem_map.shape != (H, W):
        sem_map = cv2.resize(sem_map, (W, H), interpolation=cv2.INTER_NEAREST)

    color = _colorize_label_map(sem_map)  # (H,W,3) uint8
    overlay = (alpha * img_bgr + (1 - alpha) * color).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay)
# ======================================================




def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')

    #positional arguments(esssential)
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help=
        'text prompts, including categories separated by a comma or a txt file with each line as a prompt.'
    )

    #optional arguments
    parser.add_argument('--topk',
                        default=100,
                        type=int,
                        help='keep topk predictions.') 
    #topk predictions means the number of top predictions to keep after filtering by score threshold
    parser.add_argument('--threshold',
                        default=0.1,
                        type=float,
                        help='confidence score threshold for predictions.')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device used for inference.')
    parser.add_argument('--show',
                        action='store_true',
                        help='show the detection results.')
    parser.add_argument(
        '--annotation',
        action='store_true',
        help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use mixed precision for inference.')
    parser.add_argument('--output-dir',
                        default='demo_outputs',
                        help='the directory to save outputs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def inference_detector(model,
                       image,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       output_dir='./work_dir',
                       use_amp=False,
                       show=False,
                       annotation=False):
    print(f"[Debug] Start Inference")
    img_path = image
    data_info = dict(img_id=0, img_path=img_path, texts=texts)
    print(f"[Debug] data_info : {data_info}")
    data_info = test_pipeline(data_info)
    print(f"[Debug] after test_pipeline data_info : {data_info}")
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])
    print(f"[Debug] data_batch : {data_batch}")                    

    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        print(f"output : {output}")
        pred_instances = output.pred_instances
        # [ADDED] fetch semantic map (if any)
        sem_map = None
        if hasattr(output, 'pred_sem_seg') and (output.pred_sem_seg is not None):
            sem_map = output.pred_sem_seg.get('sem_seg', None)
        print(f"[Debug] pred_instances : {pred_instances}")
        print(f"[Debug] coeffs {pred_instances.coeffs.shape}")
        print(f"[Debug] labels {pred_instances.labels.shape}")
        print(f"[Debug] masks  {pred_instances.masks.shape} sum={pred_instances.masks.sum()}")
        print(f"[Debug] scores {pred_instances.scores.shape}")
        print(f"[Debug] bboxes {pred_instances.bboxes.shape}")
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    print(f"[Debug] (filtered) coeffs {pred_instances.coeffs.shape}")
    print(f"[Debug] (filtered) labels {pred_instances.labels.shape}")
    print(f"[Debug] (filtered) masks  {pred_instances.masks.shape} sum={pred_instances.masks.sum()}")
    print(f"[Debug] (filtered) scores {pred_instances.scores.shape}")
    print(f"[Debug] (filtered) bboxes {pred_instances.bboxes.shape}")

    # === Detection visualization ===
    pred_instances = pred_instances.cpu().numpy()

    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None

    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)

    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    # label images
    #image = cv2.imread(image_path)
    img_bgr = cv2.imread(img_path)
    anno_image = img_bgr.copy()
    det_image = BOUNDING_BOX_ANNOTATOR.annotate(anno_image, detections)
    det_image = LABEL_ANNOTATOR.annotate(det_image, detections, labels=labels)
    if masks is not None:
      det_image = MASK_ANNOTATOR.annotate(det_image, detections)
    cv2.imwrite(osp.join(output_dir, osp.basename(img_path)), det_image)

    # save semantic overlay
    sem_out_path = None
    if sem_map is not None:
        stem, _ = osp.splitext(osp.basename(img_path))
        sem_out_path = osp.join(output_dir, f"{stem}_sem.png")
        # bg_id(last class) exclusion
        bg_id = len(texts) - 1
        sem_np = sem_map.detach().cpu().numpy() if torch.is_tensor(sem_map) else sem_map
        if sem_np.ndim == 3:
          sem_np = sem_np.argmax(0) if sem_np.shape[0] > 1 else sem_np[0]
        sem_np = sem_np.astype(np.int32)
        sem_np[sem_np == bg_id] = -1   # -1 : colorize skip → transparent
        save_semantic_overlay(img_bgr, sem_np, sem_out_path, alpha=0.5)

    if annotation:
        images_dict = {}
        annotations_dict = {}

        images_dict[osp.basename(img_path)] = anno_image
        annotations_dict[osp.basename(img_path)] = detections

        ANNOTATIONS_DIRECTORY = os.makedirs(r"./annotations", exist_ok=True)

        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(
            classes=texts, images=images_dict,
            annotations=annotations_dict).as_yolo(
                annotations_directory_path=ANNOTATIONS_DIRECTORY,
                min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
                max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
                approximation_percentage=APPROXIMATION_PERCENTAGE)

    if show:
        cv2.imshow('Detections', det_image)
        if sem_out_path is not None:
            sem_vis = cv2.imread(sem_out_path)
            if sem_vis is not None:
                cv2.imshow('Semantic', sem_vis)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    # init model
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    # test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(test_pipeline_cfg)

    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']] # lines = ["bus\n", "person\n"] --> [['bus'], ['person'], [' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']] # "bus, person" --> [['bus'], ['person'], [' ']]

    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # load images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.endswith('.png') or img.endswith('.jpg')
        ]
    else:
        images = [args.image]

    # reparameterize texts
    model.reparameterize(texts)
    progress_bar = ProgressBar(len(images))
    for image_path in images:
        inference_detector(model,
                           image_path,
                           texts,
                           test_pipeline,
                           args.topk,
                           args.threshold,
                           output_dir=output_dir,
                           use_amp=args.amp,
                           show=args.show,
                           annotation=args.annotation)
        progress_bar.update()
