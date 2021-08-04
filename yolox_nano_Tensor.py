from loguru import logger
import cv2
import torch
import sys
sys.path.append('D:\github\YOLOX')
from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

import argparse
import os
import time