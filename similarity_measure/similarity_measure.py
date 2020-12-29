import numpy as np
import argparse
from scipy import spatial

import sys
sys.path.append('./')
from config import SIZE_PROJECTION
from util import signature_bit


class Cosine_Measure:

    def compute_similarity(self, X, Y):
        cosine_dist = 1 - np.apply_along_axis(spatial.distance.cosine, 1, Y, X)
        return cosine_dist*100


class Euclidean_Measure:
    
    def compute_similarity(self, X, Y):
        euclidean_dist = np.apply_along_axis(spatial.distance.euclidean, 1, Y, X)
        return euclidean_dist

class Manhatan_Measure: 

    def compute_similarity(self, X, Y):
        mahatan_dist =  np.apply_along_axis(spatial.distance.cityblock, 1, Y, X)
        return mahatan_dist
# Compute IOU with hash 

class IOU_Measure:

    def __init__(self, projections_size):
        self.projections_size = projections_size

    def bitcount(self,n):
      """
      gets the number of bits set to 1
      """
      count = 0
      while n:
        count += 1
        n = n & (n-1)
      return count

    def compute_IOU(self, sig1, sig2):
      sig1 = sig1[0]
      sig2 = sig2[0]
      # print('shape sig2: ', sig2)
      # print('shape sig2: ', sig2)
      return 1 - self.bitcount(sig1^sig2)/self.projections_size

    def compute_similarity(self, X,Y):
      IOU_dist = np.apply_along_axis(self.compute_IOU, 1,Y,[X])
      return IOU_dist*100

def main(args):
    X = np.random.randint(255, size=(1, 5))
    print('Vector X: ', X)
    Y = np.random.randint(255, size=(232, 5))
    print('Vector Y: ', Y)

    Measure = None

    if args['method'] == 'cosine':
        Measure = Cosine_Measure()
    elif args['method'] == 'euclidean':
        Measure = Euclidean_Measure()
    elif args['method'] == 'manhatan':
        Measure = Manhatan_Measure()
    elif args['method'] == 'lsh_IOU':
        Measure = IOU_Measure(SIZE_PROJECTION)
    else:
      print("[ERROR]:Wrong method. Pleas enter similarity measure again!!!")
    dist = Measure.compute_similarity(X, Y)
    print(" Number distance X and matrix Y: ", dist.shape)
    # print("Distance between vector X and matrix Y: ", dist)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Methods similarity measure image.")
    parser.add_argument('-m', '--method',  default="norm2",
                        help="Method similarity measure between vector.")

    # End default optional arguments

    args = vars(parser.parse_args())
    main(args)