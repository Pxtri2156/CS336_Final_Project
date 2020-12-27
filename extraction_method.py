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

from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from six.moves.urllib.request import urlopen

import glob
from itertools import accumulate
from tensorflow.python.framework import ops

from config import MAX_DESCRIPTOR
from tqdm import tqdm

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

    return vgg16_feature.flatten()

def extrac_histogram(img):
  img = cv2.resize(img, (128,128))
  HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
  histogram_H = cv2.calcHist([HSV_img], [0], None, [256], [0 , 255])
  histogram_S = cv2.calcHist([HSV_img], [1], None, [256], [0 , 255])
  histogram_V = cv2.calcHist([HSV_img], [2], None, [256], [0 , 255])
  histogram = np.concatenate((histogram_H,histogram_S,histogram_V), 1)

  return histogram.flatten()

def deep(img, model):
  pass

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)

def extract_delf(image_tf, path_list, module_outputs, image_placeholder):
  with tf.train.MonitoredSession() as sess:
    results_dict = {}  # Stores the locations and their descriptors for each image
    for image_path in tqdm(path_list) :
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % image_path)
        results_dict[image_path] = sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})
    return results_dict

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

    elif args['method'] == 'COLOR': # color use
        feature = extrac_histogram(img)
    elif args['method'] == 'DELF':

        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.FATAL)

        m = hub.Module('https://tfhub.dev/google/delf/1')

        # The module operates on a single image at a time, so define a placeholder to
        # feed an arbitrary image in.
        image_placeholder = tf.placeholder(
            tf.float32, shape=(None, None, 3), name='input_image')

        module_inputs = {
            'image': image_placeholder,
            'score_threshold': 100.0,
            'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            'max_feature_num': 1000,
        }

        module_outputs = m(module_inputs, as_dict=True)
        image_tf = image_input_fn([args['input_path']])
        path_list = [args['input_path']]
        des_dic = extract_delf(image_tf, path_list, module_outputs, image_placeholder)
        print('Descriptor dictionary: ', des_dic.values())
    if args['method'] != "DELF":
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
