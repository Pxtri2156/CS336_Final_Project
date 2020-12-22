import numpy as np

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
    dist = X - Y
    ind_img = np.argmin(np.linalg.norm(dist, axis = 1))
    return dist

def main(args):
    X = np.random.randint(255, size=(1, 1000))
    Y = np.random.randint(255, size=(232, 1000))

    if args['norm2']:
        dis = norm2(X,Y)
    elif args['cosine']:
        pass
    elif args['euclid']:
        pass
    elif args['mahatan']:
        pass

    print("Distance between vector X and matrix Y: ", dis)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Methods similarity measure image.")
    parser.add_argument('-m', '--method',  default="norm2",
                        help="Method similarity measure between vector.")

    # End default optional arguments

    args = vars(parser.parse_args())
    main(args)