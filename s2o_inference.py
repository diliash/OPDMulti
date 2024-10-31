# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OPDFormer Training Script.

This script is a simplified version of the training script in mask2former/train_net.py.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import argparse
import datetime
import itertools
import json
import os
import sys
from glob import glob
from time import time

import cv2
import detectron2.utils.comm as comm
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import verify_results
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from tqdm import tqdm

sys.path.append("./opdformer")
# MaskFormer
from mask2former import (
    MotionVisualizer,
    add_maskformer2_config,
    add_motionnet_config,
    register_motion_instances,
)
from PIL import Image


def prediction_to_json(instances, img_id, motionstate=False):
    """
    Args:
        instances (Instances): the output of the model
        img_id (str): the image id in COCO

    Returns:
        list[dict]: the results in densepose evaluation format
    """
    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    # Prediction for MotionNet
    # mtype = instances.mtype.squeeze(axis=1).tolist()

    # 2.0.3
    if instances.has("pdim"):
        pdim = instances.pdim.tolist()
    if instances.has("ptrans"):
        ptrans = instances.ptrans.tolist()
    if instances.has("prot"):
        prot = instances.prot.tolist()

    mtype = instances.mtype.tolist()
    morigin = instances.morigin.tolist()
    maxis = instances.maxis.tolist()
    if instances.has("mextrinsic"):
        mextrinsic = instances.mextrinsic.tolist()

    if motionstate:
        mstate = instances.mstate.tolist()

    # MotionNet has masks in the annotation
    # use RLE to encode the masks, because they are too large and takes memory
    # since this evaluator stores outputs of the entire dataset
    rles = [
        mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        for mask in instances.pred_masks
    ]
    for rle in rles:
        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        # json writer which always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does).
        rle["counts"] = rle["counts"].decode("utf-8")

    results = []
    for k in range(len(instances)):
        if instances.has("pdim"):
            result = {
                "image_id": img_id,
                "category_id": classes[k] + 1,
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "pdim": pdim[k],
                "ptrans": ptrans[k],
                "prot": prot[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
            }
        elif instances.has("mextrinsic"):
            result = {
                "image_id": img_id,
                "category_id": classes[k] + 1,
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
                "mextrinsic": mextrinsic[k],
            }
        else:
            result = {
                "image_id": img_id,
                "category_id": classes[k] + 1,
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
            }
        if motionstate:
            result["mstate"] = mstate[k]
        results.append(result)
    return results

def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = "path/to/ckpt"
    cfg.merge_from_list(args.opts)
    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

    # Input format
    cfg.INPUT.FORMAT = args.input_format
    if args.input_format == "RGB":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[0:3]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[0:3]
    elif args.input_format == "depth":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[3:4]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[3:4]
    elif args.input_format == "RGBD":
        pass
    else:
        raise ValueError("Invalid input format")

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="opdformer")
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Train OPDFormer")
    parser.add_argument(
        "--config-file",
        default="opdformer/configs/opd_p_real.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output-dir",
        default=f"output-s2o/",
        metavar="DIR",
        help="path for training output",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--input-format",
        default="RGBD",
        choices=["RGB", "RGBD", "depth"],
        help="input format (RGB, RGBD, or depth)",
    )
    parser.add_argument(
        "--prob",
        required=False,
        type=float,
        default=0.9,
        help="indicating the smallest probability to visualize",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    with torch.no_grad():

        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        predictions = []
        all_outputs = []
        model_ids = np.sort([path.split("/")[-1] for path in glob(f"{args.data_path}/origin/*") if len(os.listdir(path)) > 0])
        image_id = 0
        for model_id in tqdm(model_ids):
            for i in range(3):
                image_path = f"{args.data_path}/origin/{model_id}/{model_id}.png.origin-{i}.png"
                depth_path = f"{args.data_path}/depth/{model_id}/{model_id}.png.depth-{i}.png"

                im = np.asarray(Image.open(image_path).convert("RGB"))
                depth = np.asarray(Image.open(depth_path), dtype=np.float32)[:, :, None]
                image = np.concatenate([im, depth], axis=2)
                img = torch.as_tensor(
                        np.ascontiguousarray(image.transpose(2, 0, 1))
                    )
                inputs = {"image": img, "height": 256, "width": 256}

                outputs = model([inputs])[0]
                prediction = {"image_id": image_id}
                instances = outputs["instances"].to("cpu")
                prediction["instances"] = prediction_to_json(instances, image_id)
                predictions.append(prediction)
                image_id += 1
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        os.makedirs(f"{args.output_dir}/inference", exist_ok=True)
        with open(f"{args.output_dir}/inference/coco_motion_results.json", "w+") as json_file:
            json.dump(coco_results, json_file)
