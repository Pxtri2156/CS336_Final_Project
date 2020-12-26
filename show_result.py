import argparse
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
def show(ranks, paths, scores):
  for i, rank in enumerate(ranks):
    print("Result with query {}".format(i))
    for j, candidate in enumerate(rank):
      img_path = paths[rank][j]
      print("Before replace: ", img_path)
      # postprocess image path 
      word_replace = img_path.split('/')[3]
      img_path = img_path.replace(word_replace,'storage' )
      print("After replace: ", img_path)
      img = cv2.imread(img_path)
      print(img.shape)
      cv2_imshow(img)
      score = scores[i][j] 
      print("Top: {} scores: {}\nImage query: {}".format(j+1, score, img_path))
    
def main(args):
  data = np.load(args["result_path"])
  ranks = data['ranks']
  scores = data['scores']
  paths = data['paths']
  show(ranks, paths, scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-rp', '--result_path',  default=".\data\train",
                        help="The path of the result .")

    args = vars(parser.parse_args())

    main(args)