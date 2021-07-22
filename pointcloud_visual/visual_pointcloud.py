# Author: Pannapann, Goddy, Neptd, WinnamonRoll
# python predict_pointcloud_cleaned.py --video footprints/monodepth2/gggg3.avi --monodepth2_model_name HR_Depth_K_M_1280x384 --pred_metric_depth

#import library
from __future__ import absolute_import, division, print_function
import cv2
from PIL import Image
import os
import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
import monodepth2.networks as networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist
import kitti_util
from pyntcloud import *
import pandas as pd
import plotly.graph_objects as go
import time
import matplotlib.pyplot as plt
import pyvista
count = 0
for i in range(1,1000,1):
    count = count + 1
    with open("out_file_{}.npz".format(int(count))) as f:
        a = np.load(f, allow_pickle=True)
    for i in a:
        #i.plot(initial_point_size=0.000002, backend="pyvista") #backend can be threejs pythreejs matplotlib and pyvista
        mesh = pyvista.PolyData(i)
        mesh
    print(a.shape)