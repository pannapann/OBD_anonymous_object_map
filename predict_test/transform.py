import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import transform
from skimage.transform import resize
import cv2 as cv2

palawan = imread('gggg_Moment.jpg')
#image_resized = resize(palawan, (palawan.shape[0] // 4, palawan.shape[1] // 4),
                       #anti_aliasing=True)
imshow(palawan);
area_of_interest = [(955-200, 115-50),
                    (380+100, 115-50),
                    (96, 295),
                    (1185, 295)]
area_of_projection = [(1185, 15),
                      (46+50, 15),
                      (46+50, 295),
                      (1185, 295)]

def cropped_frame(input):
    w = -300
    crop_frame = input[w:,:]
    return crop_frame

def project_planes(image, src, dst):
    x_src = [val[0] for val in src] + [src[0][0]]
    y_src = [val[1] for val in src] + [src[0][1]]
    x_dst = [val[0] for val in dst] + [dst[0][0]]
    y_dst = [val[1] for val in dst] + [dst[0][1]]
    
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    
    new_image = image.copy() 
    #print(new_image.shape)
    #new_image = np.zeros((500, 1280,3))
    projection = np.zeros_like(new_image)
    ax[0].imshow(new_image);
    ax[0].plot(x_src, y_src, 'r--')
    ax[0].set_title('Area of Interest')
    ax[1].imshow(projection)
    ax[1].plot(x_dst, y_dst, 'r--')
    ax[1].set_title('Area of Projection')
palawan = cropped_frame(palawan)
project_planes(palawan, area_of_interest, area_of_projection)


def project_transform(image, src, dst):
    x_dst = [val[0] for val in dst] + [dst[0][0]]
    y_dst = [val[1] for val in dst] + [dst[0][1]]
    
    tform = transform.estimate_transform('projective', 
                                         np.array(src), 
                                         np.array(dst))
    transformed = transform.warp(image, tform.inverse)
    #print(transformed.shape)
    plt.figure(figsize=(6,6))
    image_resized = resize(transformed, (transformed.shape[0]+200 , transformed.shape[1]),
                       anti_aliasing=True)
    #plt.imshow(transformed)
    plt.imshow(image_resized)
    plt.plot(x_dst, y_dst, 'r--')
    plt.show()
project_transform(palawan, area_of_interest, area_of_projection)
















