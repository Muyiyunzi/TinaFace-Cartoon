import argparse

import cv2
import numpy as np
import torch

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine

filename = "cartoon_gt.txt"
fo = open(filename, "w")
for i in range(42500):
    STR = "images/" + str(i) + ".jpg" + "\n"
    fo.write(STR)

    STR = "annotations/" + str(i) + ".txt"
    fi = open(STR, "r")

    lines = str(len(fi.readlines())) + "\n"
    fo.write(lines)
    fi = open(STR, "r") # traceback pointer
    s = fi.read()
    w = fo.write(s)


