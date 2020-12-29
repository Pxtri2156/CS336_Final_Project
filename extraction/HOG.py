
import cv2 
from skimage.feature import hog
import argparse

class HOG:

    def __init__(self):
        pass

    def extract(self, img ):
        resized_img = cv2.resize(img, (128, 128))
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
        
        return fd

def main(args):
    img = cv2.imread(args['input_path'])    
    extractor = HOG()
    feature = extractor.extract(img)
    print("Shape feature: ", feature.shape)
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
