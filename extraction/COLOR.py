
import cv2 
from skimage.feature import hog
import argparse
import numpy as np

class COLOR:

    def __init__(self):
        pass 

    def extract(self, img ):
        img = cv2.resize(img, (128,128))
        HSV_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        histogram_H = cv2.calcHist([HSV_img], [0], None, [256], [0 , 255])
        histogram_S = cv2.calcHist([HSV_img], [1], None, [256], [0 , 255])
        histogram_V = cv2.calcHist([HSV_img], [2], None, [256], [0 , 255])
        histogram = np.concatenate((histogram_H,histogram_S,histogram_V), 1)

        return histogram.flatten()


def main(args):
    img = cv2.imread(args['input_path'])
    extrator = None
    extractor = COLOR()
    feature = extractor.extract(img)
    feature2 = extractor.extract(img)
    print("Shape feature: ", feature.shape)
    print("Shape feature 2: ", feature2.shape)
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
