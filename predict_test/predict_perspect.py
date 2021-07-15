# Copyright Niantic 2019. Patent Pending. All rights reserved.
# python predict_perspect.py --video monodepth2/gggg.avi --monodepth2_model_name mono+stereo_640x192 --pred_metric_depth
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from footprints.model_manager import ModelManager
from footprints.utils import sigmoid_to_depth, download_model_if_doesnt_exist, pil_loader, MODEL_DIR
from footprints.utils import download_model_if_doesnt_exist as f_model_load
import torch
from torchvision import transforms, datasets
import monodepth2.networks as networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist
from monodepth2.evaluate_depth import STEREO_SCALE_FACTOR
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from skimage import transform

MODEL_HEIGHT_WIDTH = {
    "kitti": (192, 640),
    "matterport": (512, 640),
    "handheld": (256, 448),
}

class InferenceManager:

    def __init__(self, model_name, use_cuda, save_visualisations=True):

        f_model_load(model_name)
        model_load_folder = os.path.join(MODEL_DIR, model_name)
        self.model_manager = ModelManager(is_inference=True, use_cuda=use_cuda)
        self.model_manager.load_model(weights_path=model_load_folder)
        self.model_manager.model.eval()

        self.use_cuda = use_cuda
        self.colormap = plt.get_cmap('plasma', 256)  # for plotting
        self.resizer = transforms.Resize(MODEL_HEIGHT_WIDTH[model_name],
                                         interpolation=Image.ANTIALIAS)
        self.totensor = transforms.ToTensor()

        #self.save_dir = save_dir
        #os.makedirs(os.path.join(save_dir,  "outputs"), exist_ok=True)
        #self.save_visualisations = save_visualisations
        #if save_visualisations:
            #os.makedirs(os.path.join(save_dir,  "visualisations"), exist_ok=True)

    def _load_and_preprocess_image(self, frame):
        """Load an image, resize it, convert to torch and if needed put on GPU
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(img)
        preprocessed_image = self.resizer(original_image)
        preprocessed_image = self.totensor(preprocessed_image)
        preprocessed_image = preprocessed_image[None, ...]
        if self.use_cuda:
            preprocessed_image = preprocessed_image.cuda()
        return original_image, preprocessed_image


    def predict_hidden_depth(self, frame):
        """Use the model to predict hidden depth
        """
        original_image, preprocessed_image = self._load_and_preprocess_image(frame)
        pred = self.model_manager.model(preprocessed_image)
        pred = pred['1/1'].data.cpu().numpy().squeeze(0)

        hidden_ground = cv2.resize(pred[1], original_image.size) > 0.5
        hidden_depth = cv2.resize(sigmoid_to_depth(pred[3]), original_image.size)
        original_image = np.array(original_image) / 255.0

        # normalise the relevant parts of the depth map and apply colormap
        _max = hidden_depth[hidden_ground].max()
        _min = hidden_depth[hidden_ground].min()
        hidden_depth = (hidden_depth - _min) / (_max - _min)
        depth_colourmap = self.colormap(hidden_depth)[:, :, :3]  # ignore alpha channel

            # create and save visualisation image
        hidden_ground = hidden_ground[:, :, None]
        # visualisation = original_image * (1 - hidden_ground) + depth_colourmap * hidden_ground
        return hidden_ground,depth_colourmap #,(visualisation[:, :, ::-1] * 255).astype(np.uint8),


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
    return parser.parse_args()


def preprocess(frame,feed_width, feed_height):
    #input_image = pil.open(image_path).convert('RGB')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(img)
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    output_image = transforms.ToTensor()(input_image).unsqueeze(0)
    return output_image,original_width, original_height
'''
def save_colormap_depthimage():
              # Saving colormapped depth image
  	disp_resized_np = disp_resized.squeeze().cpu().numpy()
  	vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
'''


def test_simple(args,frame):
    """Function to predict for a single image or folder of images
    """
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
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
		#AI Start
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

            # Calculating metric depth
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            return opencvImage


def cropped_frame(input):
    w = -300
    crop_frame = input[w:,:]
    return crop_frame

def skimage2opencv(src):
    src *= 255
    src = src.astype(int)
    cv2.cvtColor(src.astype(np.uint8),cv2.COLOR_RGB2BGRA)
    return src

def opencv2skimage(src):
    cv2.cvtColor(src.astype(np.uint8),cv2.COLOR_BGRA2RGB)
    src = src.astype(np.float32)
    src /= 255
    return src
            
            
if __name__ == '__main__':
    args = parse_args()
    inference_manager = InferenceManager(
    model_name=args.footprint_model,
    use_cuda=torch.cuda.is_available() and not args.no_cuda,
    save_visualisations=not args.no_save_vis)
    #save_dir=args.save_dir)
    cap = cv2.VideoCapture(args.video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height-300))
    first = True
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            hidden_ground,depth_colourmap = inference_manager.predict_hidden_depth(frame)
            depth_array = test_simple(args,frame)
            #background_image,_ = inference_manager._load_and_preprocess_image( depth_array)
            #depth_array = cv2.cvtColor(depth_array, cv2.COLOR_RGB2BGR)
            ground_color = 0.2*depth_colourmap * hidden_ground
            frame = depth_array*(1 - hidden_ground) + (ground_color[:,:,] * 255).astype(np.uint8)
            
            frame = cropped_frame(frame)
            img = frame.astype(np.uint8)
            
            cv2.circle(img, (380, 115), 5, (0, 0, 255), -1)
            cv2.circle(img, (555+400, 115), 5, (0, 0, 255), -1)
            cv2.circle(img, (910+275, 295), 5, (0, 0, 255), -1)
            cv2.circle(img, (46+50, 295), 5, (0, 0, 255), -1)
            pts1 = np.float32([[380, 115], [555+400, 115],[46+50, 295],[910+275, 295]])
            pts2 = np.float32([[46+50, 115], [910+275, 115],[46+50, 295],[910+275, 295]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            result = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
            if first:
                out = cv2.VideoWriter('outperspect1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (img.shape[1], img.shape[0]),1)
                first=False
            out.write(result)
            cv2.imshow("Image", img)
            cv2.imshow("Perspective transformation", result)
           
            '''frame = opencv2skimage(frame)
            gray_painting = rgb2gray(frame)
            binarized = gray_painting>0.3
            label_image = label(binarized)
            frame = label2rgb(label_image, image=binarized, bg_label=0,colors=["white"])
            frame =skimage2opencv(frame)'''
            
            #frame = (ground_color[:,:,] * 255).astype(np.uint8)
            #cv2.imshow('frame',frame.astype(np.uint8))
            #print(frame.shape[0])
            #print(frame.shape[1])
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
cap.release()
out.release()

cv2.destroyAllWindows()      
