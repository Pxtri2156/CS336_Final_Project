import cv2 
import argparse
import numpy as np
from glob import glob

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

from tqdm import tqdm


class DeepDELF:

    def __init__(self,input_path):

        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.FATAL)

        m = hub.Module('https://tfhub.dev/google/delf/1')

        # The module operates on a single image at a time, so define a placeholder to
        # feed an arbitrary image in.
        self.image_placeholder = tf.placeholder(
            tf.float32, shape=(None, None, 3), name='input_image')

        module_inputs = {
            'image': self.image_placeholder,
            'score_threshold': 100.0,
            'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            'max_feature_num': 1000,
        }

        self.module_outputs = m(module_inputs, as_dict=True)
        self.image_tf = self.image_input_fn(glob(input_path + '/*'))
        self.path_list = glob(input_path + '/*')

    def extract(self):
        with tf.train.MonitoredSession() as sess:
          results_dict = {}  # Stores the locations and their descriptors for each image
          for image_path in tqdm(self.path_list) :
              image = sess.run(self.image_tf)
              print('Extracting locations and descriptors from %s' % image_path)
              results_dict[image_path] = sess.run(
                  [self.module_outputs['locations'], self.module_outputs['descriptors']],
                  feed_dict={self.image_placeholder: image})
          return results_dict

    def image_input_fn(self, image_files):
        filename_queue = tf.train.string_input_producer(
            image_files, shuffle=False)
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        image_tf = tf.image.decode_jpeg(value, channels=3)
        return tf.image.convert_image_dtype(image_tf, tf.float32)



def main(args):
  
    path = args['input_path']
    extrator = None
    extractor = DeepDELF(path)
    results_dict = extractor.extract()
    results_dict2 = extractor.extract()
    print("Shape feature: ", results_dict.keys())
    print("Shape feature 2: ",results_dict2.keys())
def args_parser():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_path',  required=True,
                        help="The path of the input image.")
    return vars(parser.parse_args())

if __name__ == "__main__":

    args = args_parser()
    # End default optional arguments
    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}:\n|{:<30}|".format("Image path", args['input_path']).center(100))
    print(str("-"*63).center(100))

    main(args)  
