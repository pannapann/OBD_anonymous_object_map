# Author: Pannapann, Goddy, Neptd, WinnamonRoll
# python predict_pointcloud_cleaned.py --video footprints/monodepth2/gggg3.avi --monodepth2_model_name HR_Depth_K_M_1280x384 --pred_metric_depth
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


STEREO_SCALE_FACTOR = 5.4

MODEL_HEIGHT_WIDTH = {
    "kitti": (192, 640),
    "matterport": (512, 640),
    "handheld": (256, 448),
}
def parse_args():
    parser = argparse.ArgumentParser(
        description='Complicate function to run monodepth+footprints on video.')

    parser.add_argument('--video', type=str,
                        help='path to a video', required=True)
    parser.add_argument('--monodepth2_model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320",
                            "Lite_HR_Depth_K_T_1280x384",
                            "HR_Depth_CS_K_MS_640x192",
                            "HR_Depth_K_M_1280x384"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument('--footprint_model', type=str,
                        help='name of a pretrained model to use',
                        choices=["kitti", "matterport", "handheld"],
                        default="kitti")
    parser.add_argument("--no_save_vis",
                        help='if set, disables visualisation saveing',
                        action='store_true')
    parser.add_argument("--save_dir", type=str,
                        help='where to save npy and visualisations to',
                        default="predictions")
    parser.add_argument('--calib_dir', type=str,
                        default=r"./footprints/calib")
    return parser.parse_args()

#preprocess intitial photo to fit the torch tensor 
def preprocess(frame,feed_width, feed_height):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(img)
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
    output_image = transforms.ToTensor()(input_image).unsqueeze(0)
    return output_image,original_width, original_height

#project display point(depth array) to cloud point
def project_disp_to_points(calib, disp, max_high,frame):
    disp[disp < 0] = 0
    baseline = 0.54
    mask = disp > 0
    depth =disp
    depth = depth[0][0]
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    blue = frame[:,:,0].reshape(-1)
    print(blue.shape)
    green = frame[:,:,1].reshape(-1)
    print(green.shape)
    red = frame[:,:,2].reshape(-1)
    print(red.shape)
    rgb = np.stack([red,green,blue])
    rgb = np.swapaxes(rgb,0,1)
    print(rgb.shape)
    #rgb = rgb.reshape((-1,3))
    points = points.reshape((3, -1))
    print(points.shape)
    points = points.T
    points = points[mask.reshape(-1)]
    cloud = calib.project_image_to_velo(points)
    cloud.shape
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid], rgb[valid]
#paint the point
def paint_points(points, color):
    new_pts = np.zeros([points.shape[0],6])
    new_pts[:,:3] = points
    new_pts[:, 3:] = new_pts[:, 3:] + color
    return new_pts

# def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
#         '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
#             00135     '''
#          # make it 2d (even if height will be 1)
#     cloud_arr = np.atleast_2d(cloud_arr)
    
#     cloud_msg = PointCloud2()
    
#     if stamp is not None:
#         cloud_msg.header.stamp = stamp
#     if frame_id is not None:
#         cloud_msg.header.frame_id = frame_id
#     cloud_msg.height = cloud_arr.shape[0]
#     cloud_msg.width = cloud_arr.shape[1]
#     cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
#     cloud_msg.is_bigendian = False # assumption
#     cloud_msg.point_step = cloud_arr.dtype.itemsize
#     cloud_msg.row_step = cloud_msg.point_step*cloud_arr.shape[1]
#     cloud_msg.is_dense = all([np.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])
#     cloud_msg.data = cloud_arr.tostring()
#     return cloud_msg


def predict_depth(args,frame,feed_width, feed_height,device):

    with torch.no_grad():
            #preprocess
            input_image, original_width, original_height = preprocess(frame,feed_width, feed_height)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            
            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            __,new_depth = disp_to_depth(disp_resized_np, 0.1, 100)
            metric_depth = STEREO_SCALE_FACTOR * new_depth
            return metric_depth
          
            
if __name__ == '__main__':
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    first = True
    calib_file = '{}/{}.txt'.format(args.calib_dir, 'calib')
    calib = kitti_util.Calibration(calib_file)
    assert args.monodepth2_model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.monodepth2_model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.monodepth2_model_name)
    model_path = os.path.join("models", args.monodepth2_model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc,strict=False)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    depth_decoder = networks.HRDepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

		#AI Start
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            #Predict depth and traversable Path
            depth_array = predict_depth(args,frame,feed_width, feed_height,device)
            disp_map = torchvision.transforms.ToTensor()(depth_array).unsqueeze(0).cpu().numpy()
            disp_map = (disp_map*256).astype(np.uint16)/256.
            lidar,colors = project_disp_to_points(calib, disp_map, 10, frame)
            # pad 1 in the indensity dimension
            lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
            lidar = lidar.astype(np.float32)
            #lidar.tofile('{}/{}.bin'.format(args.save_dir, predix))
            points = lidar.reshape((-1, 4))[:,:3]
            print("Painting points")
            pd_points = pd.DataFrame(paint_points(points,colors), columns=['x','y','z','red','green','blue'])
            cloud = PyntCloud(pd_points)
            cloud.plot(initial_point_size=0.000002, backend="matplotlib")
            # print("plotting")
            # marker_data = go.Scatter3d(
            #     x=pd_points['x'].to_numpy(),
            #     y=pd_points['y'].to_numpy(), 
            #     z=pd_points['z'].to_numpy(), 
            #     marker=dict(color=[f'rgb({r}, {g}, {b})' for r,g,b in zip(pd_points['red'].to_numpy(),pd_points['green'].to_numpy(),pd_points['blue'].to_numpy())],
            #            size=1), 
            #     opacity=0.8, 
            #     mode='markers'
            # )
            # fig=go.Figure(data=marker_data)
            # fig.show()
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
        
cap.release()
cv2.destroyAllWindows()      
 
