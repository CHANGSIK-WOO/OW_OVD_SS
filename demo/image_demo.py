# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import argparse
import os.path as osp
import numpy as np

import torch
from mmengine.config import Config, DictAction
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmengine.utils import ProgressBar
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg

import supervision as sv

# ===================== Utils for Semantic Visualization ======================
def _build_palette(num_classes: int, seed: int = 42):
    """Create random BGR palette of shape (C,3) uint8."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

def _colorize_sem_map(sem_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map (H,W) int labels to (H,W,3) BGR color image with given palette."""
    sem_map = np.asarray(sem_map).squeeze()
    assert sem_map.ndim == 2, f"sem_map must be 2D, got {sem_map.shape}"
    sem_map_clamped = np.clip(sem_map, 0, len(palette) - 1)
    return palette[sem_map_clamped]  # (H, W, 3) uint8 (BGR)

def _overlay(image_bgr: np.ndarray, color_map_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend semantic color map onto original image."""
    return cv2.addWeighted(color_map_bgr, alpha, image_bgr, 1.0 - alpha, 0)
# ============================================================================

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image', help='image path, include image file or dir.')
    parser.add_argument(
        'text',
        help='text prompts, including categories separated by a comma or a txt file with each line as a prompt.',
    )
    parser.add_argument('--topk', default=100, type=int, help='keep topk predictions.')
    parser.add_argument('--threshold', default=0.1, type=float, help='confidence score threshold for predictions.')
    parser.add_argument('--device', default='cuda:0', help='device used for inference.')
    parser.add_argument('--show', action='store_true', help='show the detection results.')
    parser.add_argument('--annotation', action='store_true', help='save the annotated detection results as yolo text format.')
    parser.add_argument('--amp', action='store_true', help='use mixed precision for inference.')
    parser.add_argument('--output-dir', default='demo_outputs', help='the directory to save outputs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=(
            'override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. If the value to '
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            'Note that the quotation marks are necessary and that no white space is allowed.'
        ),
    )
    args = parser.parse_args()
    return args


def inference_detector(model,
                       image_path,
                       texts,
                       test_pipeline,
                       max_dets=100,
                       score_thr=0.3,
                       output_dir='./work_dir',
                       use_amp=False,
                       show=False,
                       annotation=False):
    # prepare pipeline
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = test_pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    # inference
    with autocast(enabled=use_amp), torch.no_grad():
        output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
        pred_sem_seg = getattr(output, 'pred_sem_seg', None)  # (H,W) LongTensor if present

    # top-k filter (instance part unchanged)
    if len(pred_instances.scores) > max_dets:
        indices = pred_instances.scores.float().topk(max_dets)[1]
        pred_instances = pred_instances[indices]

    # numpy conversion for supervision
    pred_instances_np = pred_instances.cpu().numpy()
    masks = pred_instances_np['masks'] if 'masks' in pred_instances_np else None

    detections = sv.Detections(
        xyxy=pred_instances_np['bboxes'],
        class_id=pred_instances_np['labels'],
        confidence=pred_instances_np['scores'],
        mask=masks,
    )

    labels = [f"{texts[cid][0]} {conf:0.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]

    # draw instance results (UNCHANGED behavior)
    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None, f"Fail to read image: {image_path}"
    anno_image = img_bgr.copy()

    drawn = BOUNDING_BOX_ANNOTATOR.annotate(img_bgr.copy(), detections)
    drawn = LABEL_ANNOTATOR.annotate(drawn, detections, labels=labels)
    if masks is not None:
        drawn = MASK_ANNOTATOR.annotate(drawn, detections)

    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), drawn)

    # add semantic outputs (ALWAYS if model provides pred_sem_seg)
    if (pred_sem_seg is not None) and hasattr(pred_sem_seg, 'data'):
        sem_map = pred_sem_seg.data.detach().cpu().numpy().astype(np.int32)  # (H,W)
        num_classes = max(1, len(texts))  # use prompt count as palette size
        palette = _build_palette(num_classes=num_classes)
        color_map_bgr = _colorize_sem_map(sem_map, palette)
        overlay = _overlay(anno_image, color_map_bgr, alpha=0.5)

        stem, _ = osp.splitext(osp.basename(image_path))
        cv2.imwrite(osp.join(output_dir, f"{stem}_sem.png"), color_map_bgr)
        cv2.imwrite(osp.join(output_dir, f"{stem}_overlay.png"), overlay)

        if show:
            cv2.imshow('Detections', drawn)
            cv2.imshow('SemMap', color_map_bgr)
            cv2.imshow('Overlay', overlay)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
    else:
        if show:
            cv2.imshow('Detections', drawn)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()

    # optional: export YOLO-format annotations (UNCHANGED)
    if annotation:
        images_dict = {osp.basename(image_path): anno_image}
        annotations_dict = {osp.basename(image_path): detections}

        os.makedirs("./annotations", exist_ok=True)
        MIN_IMAGE_AREA_PERCENTAGE = 0.002
        MAX_IMAGE_AREA_PERCENTAGE = 0.80
        APPROXIMATION_PERCENTAGE = 0.75

        sv.DetectionDataset(
            classes=texts,
            images=images_dict,
            annotations=annotations_dict
        ).as_yolo(
            annotations_directory_path="./annotations",
            min_image_area_percentage=MIN_IMAGE_AREA_PERCENTAGE,
            max_image_area_percentage=MAX_IMAGE_AREA_PERCENTAGE,
            approximation_percentage=APPROXIMATION_PERCENTAGE
        )


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # init model
    cfg.load_from = args.checkpoint
    model = init_detector(cfg, checkpoint=args.checkpoint, device=args.device)

    # init test pipeline
    test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
    test_pipeline = Compose(test_pipeline_cfg)

    # parse texts
    if args.text.endswith('.txt'):
        with open(args.text) as f:
            lines = f.readlines()
        texts = [[t.rstrip('\r\n')] for t in lines] + [[' ']]
    else:
        texts = [[t.strip()] for t in args.text.split(',')] + [[' ']]

    # output dir
    output_dir = args.output_dir
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    # collect images
    if not osp.isfile(args.image):
        images = [
            osp.join(args.image, img) for img in os.listdir(args.image)
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
    else:
        images = [args.image]

    # reparameterize texts (YOLO-World style)
    model.reparameterize(texts)

    # run
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
