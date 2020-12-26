import argparse
import numpy as np
def show(ranks, paths, scores):
  for i, rank in enumerate(ranks):
    print("Result with query {}".format(i))
    for j, candidate in enumerate(rank):
      img_path = paths[rank][j]
      print(img_path)
      score = scores[i][j] 
      print("Top: {} scores: {}\nImage query: {}".format(j+1, score, img_path))
    break
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