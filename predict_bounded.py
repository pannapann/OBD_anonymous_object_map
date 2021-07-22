# Author: Pannapann, Goddy, Neptd, WinnamonRoll
# python predict_bounded.py --video footprints/monodepth2/gggg3.avi --monodepth2_model_name HR_Depth_K_M_1280x384 --pred_metric_depth
from __future__ import absolute_import, division, print_function
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from footprints.model_manager import ModelManager
from footprints.utils import sigmoid_to_depth, download_model_if_doesnt_exist, pil_loader, MODEL_DIR
from footprints.utils import download_model_if_doesnt_exist as f_model_load
import torch
import torchvision
from torchvision import transforms
import monodepth2.networks as networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist
STEREO_SCALE_FACTOR = 5.4
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

        # create hidden ground and save visualisation image
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


def test_simple(args,frame):
    """Function to predict depth from frame with given model
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
            origin_metric_depth = metric_depth

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            __,new_depth = disp_to_depth(disp_resized_np, 0.1, 100)
            metric_depth = STEREO_SCALE_FACTOR * new_depth
            return opencvImage,metric_depth,origin_metric_depth



def cropped_frame(input):
    w = -300
    crop_frame = input[w:,:]
    return crop_frame
            
            
if __name__ == '__main__':
    args = parse_args()
    inference_manager = InferenceManager(
    					model_name=args.footprint_model,
    					use_cuda=torch.cuda.is_available() and not args.no_cuda,
    					save_visualisations=not args.no_save_vis)
    
    cap = cv2.VideoCapture(args.video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    first = True
	
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            original_frame = cropped_frame(frame)
			
            #Predict depth and traversable Path
            hidden_ground,depth_colourmap = inference_manager.predict_hidden_depth(frame)
            depth_image,depth_array,original_metric_depth = test_simple(args,frame)
            test = torchvision.transforms.ToTensor()(depth_array).unsqueeze(0).cpu().numpy()
            np.save("depthNumpy",test)
			
            #Apply Hidden ground to predicted frame
            depth_array = depth_array[240:540:]
            ground_color = 0*depth_colourmap * hidden_ground
            frame = depth_image*(1 - hidden_ground) + (ground_color[:,:,] * 255).astype(np.uint8)
            
            #Cropped and convert to grayscale
            frame = cropped_frame(frame)
            im = frame.astype(np.uint8)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			
            #Detect contour of the image
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
			
            # Draw Contours and put depth on the image
            for c in contours:
                rect = cv2.boundingRect(c)
                if rect[2]*rect[3] < 1000 or rect[2]*rect[3] > 0.5*1280*300 : continue
                print(cv2.contourArea(c))
                x,y,w,h = rect
                cv2.rectangle(original_frame,(x,y),(x+w,y+h),(0,255,0),2)
                object_range = depth_array.min()
                cv2.putText(original_frame,"Object Detected range = {range:.2f}".format(range=object_range),(x+w+10,y+h),0,0.3,(0,255,0))

                #Start video writer
                if first:
                    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame.shape[1],frame.shape[0]),1)
                    first=False
                out.write(original_frame)
			
            #Show output via opencv
            cv2.imshow("Show",original_frame)
            cv2.imshow("Colorized",im)
			
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print(depth_array.shape)
                print(test.shape)
                break
        else:
            break

cap.release()
out.release()

cv2.destroyAllWindows()      
 
