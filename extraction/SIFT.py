
import cv2 
from skimage.feature import hog
import argparse

class SIFT:

    def __init__(self):
        self.sift =  cv2.xfeatures2d.SIFT_create()

    def extract(self, img ):
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors

def main(args):
    img = cv2.imread(args['input_path'])
    extrator = None
    extractor = SIFT()
    keypoints, descriptors = extractor.extract(img)
    feature = descriptors
    keypoints2, descriptors2 = extractor.extract(img)
    feature2 = descriptors2
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
