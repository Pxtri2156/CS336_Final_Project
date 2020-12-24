import numpy as np
import argparse
from scipy import spatial

from config import SIZE_PROJECTION
from util import signature_bit
'''
X.shape = (1,m)
Y.shape = (n,m)
'''

def cosine(X, Y):
    cosine_dist = 1 - np.apply_along_axis(spatial.distance.cosine, 1, Y, X)
    return cosine_dist

def euclidean(X,Y):
    euclidean_dist = np.apply_along_axis(spatial.distance.euclidean, 1, Y, X)
    return euclidean_dist

def manhatan(X,Y):
    mahatan_dist = np.apply_along_axis(spatial.distance.cityblock, 1, Y, X)
    return mahatan_dist

def norm2(X,Y):
    sub_dist = X - Y
    norm2_dist = np.linalg.norm(sub_dist, axis = 1)
    return norm2_dist

# Compute IOU with hash 

def bitcount(n):
	"""
	gets the number of bits set to 1
	"""
	count = 0
	while n:
		count += 1
		n = n & (n-1)
	return count

def compute_IOU(sig1, sig2, projections_size):
  sig1 = sig1[0]
  sig2 = sig2[0]
  return 1 - bitcount(sig1^sig2)/projections_size

def hash_IOU(X,Y, projections_size):
  IOU_dist = np.apply_along_axis(compute_IOU, 1,Y,[X]  ,projections_size)
  return IOU_dist

def main(args):
    X = np.random.randint(255, size=(1, 5))
    print('Vector X: ', X)
    Y = np.random.randint(255, size=(232, 5))
    print('Vector Y: ', Y)


    if args['method'] == 'norm2':
        dist = norm2(X,Y)
    elif args['method'] == 'cosine':
        dist = cosine(X,Y)
    elif args['method'] == 'euclidean':
        dist = euclidean(X,Y)
    elif args['method'] == 'manhatan':
        dist = manhatan(X,Y)
    elif args['method'] == 'lsh_IOU':
        dist = hash_IOU(X,Y, SIZE_PROJECTION)
    else:
      print("[ERROR]:Wrong method. Pleas enter similarity measure again!!!")

    print(" Number distance X and matrix Y: ", dist.shape)
    print("Distance between vector X and matrix Y: ", dist)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Methods similarity measure image.")
    parser.add_argument('-m', '--method',  default="norm2",
                        help="Method similarity measure between vector.")

    # End default optional arguments

    args = vars(parser.parse_args())
    main(args)