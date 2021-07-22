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
import open3d as o3d
count = 0
for i in range(1,1000,1):
    count = count + 1
    #cloud = PyntCloud.from_file("out_fil_{}.ply".format(int(count)))
    #converted_triangle_mesh = cloud.to_instance("open3d", mesh=True)
    pcd = o3d.io.read_point_cloud("out_fil_{}.ply".format(int(count)))
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
    #with open("out_file_{}.npz".format(int(count))) as f:
    # cloud = PyntCloud.from_file("out_file_{}.npz".format(int(count)), allow_pickle=True)
    # print(1)
  