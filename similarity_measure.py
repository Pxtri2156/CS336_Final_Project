import numpy as np
import argparse
'''
X.shape = (1,m)
Y.shape = (n,m)
'''

def cosine(X, Y):
    pass

def euclid(X,Y):
    pass

def mahatan(X,Y):
    pass

def norm2(X,Y):
    sub_dist = X - Y
    norm2_dist = np.linalg.norm(sub_dist, axis = 1)
    return norm2_dist

def main(args):
    X = np.random.randint(255, size=(1, 1000))
    Y = np.random.randint(255, size=(232, 1000))

    if args['method'] == 'norm2':
        dis = norm2(X,Y)
    elif args['method'] == 'cosine':
        pass
    elif args['method'] == 'euclid':
        pass
    elif args['method'] == 'mahatan':
        pass
    else:
      print("Wrong method. Please enter similarity measure again")

    print(" Number distance X and matrix Y: ", dis.shape)
    print("Distance between vector X and matrix Y: ", dis)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Methods similarity measure image.")
    parser.add_argument('-m', '--method',  default="norm2",
                        help="Method similarity measure between vector.")

    # End default optional arguments

    args = vars(parser.parse_args())
    main(args)