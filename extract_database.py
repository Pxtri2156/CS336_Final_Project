import argparse
import numpy as np
from tqdm import tqdm
import os
import cv2

from extraction.COLOR import COLOR
from extraction.DELF import DeepDELF
from extraction.SIFT import SIFT
from extraction.SURF import SURF
from extraction.VGG16 import DeepVGG16
from extraction.HOG import HOG 
from config import SIZE_PROJECTION, RANDOM_SEED
from util import signature_bit
from glob import glob
import json

def extract_database(input_path, method, LSH):
    features = [] # save feature
    path_list = [] # save path of each image
    if method != 'DELF':

        extractor = None
        if method == "COLOR":
            extractor = COLOR()
        elif method == 'HOG':
            extractor = HOG()
        elif method == "SIFT":
            pass
        elif method == "SURF":
            pass
        elif method == "VGG16":
            extractor = DeepVGG16()
        elif method == "facenet":
            pass 

        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("\n[INFO] Processing: img: {} method: {}, use LSH: {} \npath_img: {}".format( \
            img_name, method,LSH, img_path))
            img = cv2.imread(img_path)
            feature = extractor.extract(img)
            features.append(feature)
            path_list.append(img_path)

        features = np.array(features)
    elif args['method'] == "DELF":
      extractor = DeepDELF(input_path)
      des_dic = extractor.extract()
      path_list = list(des_dic.keys())
      features = des_dic 
      print("Key feature: ", features.keys())
      print("feature: ", type(features))

    else:
       print("[ERROR]:Wrong method,  Pleas enter extract method again!!!")
        
    projections = None 
    if LSH == True:
      print('active LSH')
      projections = np.random.randn(SIZE_PROJECTION,features.shape[1])
      features = np.apply_along_axis(signature_bit,1,features,projections)
      features = features.reshape(-1, 1)

    return features, path_list, projections
        
def save_feature(features, path_list, output_path, method, LSH, k = SIZE_PROJECTION, projections = None ):
    name_save_file = method + ".npz"
    if LSH == True:
      print('active LSH')
      name_save_file = str(k) + '_LSH_' + name_save_file   
    save_path = os.path.join(output_path, name_save_file)
    np.savez_compressed(save_path, features=features, paths = path_list, projections=projections)
  
def save_dic_DELF(dics, path_list, output_path, method):
      name_save_file = method + ".json"
      paths_file = method + ".npz"

      save_path = os.path.join(output_path, name_save_file)
      paths_path =  os.path.join(output_path, paths_file)
      
      save_file = open(save_path, 'w')
      print("Before save: ", type(dics))
      json.dump(dics, save_file )
      np.savez_compressed(paths_path, paths = path_list)
      
def main(args):

    np.random.seed(RANDOM_SEED)
    # extract feature
    print("[INFO] Extracting  {} feature for dataset".format(args["method"]))
    features, path_list, projections = extract_database(args['input_folder'], args['method'], args['LSH'])
    try:
        print('len features',features.shape)
    except: 
        print("Using DELF, len of dictionary", len(features) )
    # save feature 

    print("[INFO]: Begin save feature")
    # if args['method'] != 'DELF':
    save_feature(features,path_list,  args['output_folder'], args['method'], args['LSH'],SIZE_PROJECTION, projections)
    # else:
    #     print('Acitave DELF')
    #     save_dic_DELF(features, path_list, args['output_folder'],args['method'])
    print("[INFO]: Saved feature")


def args_parse():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_folder',  default=".\data\train",
                        help="The path of the input image folder.")
    parser.add_argument('-o', '--output_folder', default=".\feature\SIFT",
                        help="The path of the output feature folder")
    parser.add_argument('-m', '--method', default="SIFT",
                        help="Method to extrac feature. We can choose such as: sift, hog, vgg16....")
    parser.add_argument('-lsh', '--LSH', type = int, default = 0,
                        help="Use Locality-Sensitive Hasing " )
    # End default optional arguments

    return vars(parser.parse_args())

if __name__ == "__main__":
    
    args = args_parse()
    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Input path", args['input_folder']).center(100))
    print("|{:<30}|{:<30}|".format("Output path", args['output_folder']).center(100))
    print("|{:<30}|{:<30}|".format("Method path", args['method']).center(100))
    print("|{:<30}|{:<30}|".format("Use LSH", args['LSH']).center(100))

    print(str("-"*63).center(100))

    main(args)
