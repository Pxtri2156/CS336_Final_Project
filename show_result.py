import argparse
import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
def show(ranks, paths, scores, replace, img_folder, query_name):
  for i, rank in enumerate(ranks):
    print("Result with query {}".format(query_name[i]))
    query_img = cv2.imread(query_name[i])
    cv2_imshow(query_img)
    for j, candidate in enumerate(rank):
      score = scores[i][j] 
      print("Top: {} scores: {}".format(j+1, score))
      img_path = paths[rank][j]  
      print("Ori path: ", img_path)
      if replace == 1 :
      # postprocess image path
          img_file = img_path.split("/")[-1]
          img_path = os.path.join(img_folder, img_file)
          print("After replace: ", img_path)
      
      img = cv2.imread(img_path)
      print(img.shape)
      cv2_imshow(img)
      
    
def main(args):
  data = np.load(args["result_path"])
  ranks = data['ranks']
  scores = data['scores']
  paths = data['paths']
  query_name = data['query_name']
  replace = args['replace_path']
  show(ranks, paths, scores, replace, args['image_folder'], query_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Methods extract image.")

    parser.add_argument('-rp', '--result_path',  default=".\data\train",
                        help="The path of the result .")

    parser.add_argument('-r', '--replace_path',  default= 1, type = int,
                        help=" 1 is replace result image path , 0 not replace path result image .")
    parser.add_argument('-i', '--image_folder',  default= './data',
                        help=" 1 is replace result image path , 0 not replace path result image .")

    args = vars(parser.parse_args())

    main(args)