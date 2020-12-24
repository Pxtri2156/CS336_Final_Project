import numpy as np
import numpy.core.multiarray 
import cv2
import os
import argparse

from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

import tensorflow as tf
from PIL import ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import MAX_DESCRIPTOR


def extract_sift(img, sift):
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def extract_hog(img):
    resized_img = cv2.resize(img, (128, 128))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd, hog_image

def extract_surf(img, surf):
    # surf = cv2.xfeatures2d.SURF_create()
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = surf.detectAndCompute(img,None)
    return keypoints, descriptors 

def extract_vgg16(img, model):
    img = cv2.resize(img, (224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    vgg16_feature = model.predict(img_data)

    return vgg16_feature

def color(img):
  pass 

def deep(img, model):
  pass
    

def main(args):
    img = cv2.imread(args['input_path'])

    if args['method'] == 'SIFT':
        sift =  cv2.xfeatures2d.SIFT_create()
        keypoints, des = extract_sift(img, sift)
        feature = des
    elif args['method'] == 'HOG':
        fd, hog_image = extract_hog(img)
        feature = fd
    elif args['method'] == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, des = extract_surf(img, surf)
        feature = des
    elif args['method'] == 'VGG16':
        model = VGG16(weights='imagenet', include_top=True)
        model.summary()
        feature = extract_vgg16(img, model)
    print('Shape {} feature: {}'.format(args['method'],feature.shape))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_path',  required=True,
                        help="The path of the input image.")
    parser.add_argument('-m', '--method', default="SIFT",
                        help="Method to extrac feature. We can choose such as: sift, hog, vgg16....")
    # End default optional arguments

    args = vars(parser.parse_args())

    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}:\n|{:<30}|".format("Image path", args['input_path']).center(100))
    print("|{:<30}|{:<30}|".format("Method", args['method']).center(100))

    print(str("-"*63).center(100))

    main(args)
