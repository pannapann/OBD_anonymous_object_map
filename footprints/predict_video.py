# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Footprints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import cv2
import glob
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from footprints.model_manager import ModelManager
from footprints.utils import sigmoid_to_depth, download_model_if_doesnt_exist, pil_loader, MODEL_DIR


MODEL_HEIGHT_WIDTH = {
    "kitti": (192, 640),
    "matterport": (512, 640),
    "handheld": (256, 448),
}


class InferenceManager:

    def __init__(self, model_name, save_dir, use_cuda, save_visualisations=True):

        download_model_if_doesnt_exist(model_name)
        model_load_folder = os.path.join(MODEL_DIR, model_name)
        self.model_manager = ModelManager(is_inference=True, use_cuda=use_cuda)
        self.model_manager.load_model(weights_path=model_load_folder)
        self.model_manager.model.eval()

        self.use_cuda = use_cuda
        self.colormap = plt.get_cmap('plasma', 256)  # for plotting
        self.resizer = transforms.Resize(MODEL_HEIGHT_WIDTH[model_name],
                                         interpolation=Image.ANTIALIAS)
        self.totensor = transforms.ToTensor()

        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir,  "outputs"), exist_ok=True)
        self.save_visualisations = save_visualisations
        if save_visualisations:
            os.makedirs(os.path.join(save_dir,  "visualisations"), exist_ok=True)

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

    def predict(self, video):
        """Use the model to predict for a single image and save results to disk
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
        visualisation = original_image * (1 - hidden_ground) + depth_colourmap * hidden_ground

        return (visualisation[:, :, ::-1] * 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
    description='Simple prediction from a footprints model.')

    parser.add_argument('--video', type=str,
        help='path to a video', required=True)
    parser.add_argument('--model', type=str,
        help='name of a pretrained model to use',
        choices=["kitti", "matterport", "handheld"])
    parser.add_argument("--no_cuda",
        help='if set, disables CUDA',
        action='store_true')
    parser.add_argument("--no_save_vis",
        help='if set, disables visualisation saveing',
        action='store_true')
    parser.add_argument("--save_dir", type=str,
        help='where to save npy and visualisations to',
        default="predictions")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    inference_manager = InferenceManager(
    model_name=args.model,
    use_cuda=torch.cuda.is_available() and not args.no_cuda,
    save_visualisations=not args.no_save_vis,
    save_dir=args.save_dir)
    cap = cv2.VideoCapture(args.video)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = inference_manager.predict(frame)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
out.release()

cv2.destroyAllWindows()   