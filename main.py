import cv2
import os

from tqdm import tqdm

from similarity_measure import*
from extraction_method import *
from config import SIZE_PROJECTION

def compute_similarity(X, Y, method):

    if method == 'norm2':
        dist = norm2(X,Y)
    elif method == 'cosine':
        dist = cosine(X,Y)
    elif method == 'euclidean':
        dist = euclidean(X,Y)
    elif method == 'manhatan':
        dist = manhatan(X,Y)
    elif method == 'lsh_IOU':
        dist = hash_IOU(X,Y, SIZE_PROJECTION)
    else:
      print("[ERROR]:Wrong method. Pleas enter similarity measure again!!!")

    return dist
def retrieval_image(feature_method, similarity_method, input_path, features_storage ):
    ranks = []
    if feature_method == 'SIFT':
        sift =  cv2.xfeatures2d.SIFT_create()
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, feature_method, img_path))
            img = cv2.imread(img_path)
            keypoints, des = extract_sift(img, sift)
            feature = des
            features.append(feature)

    elif feature_method == 'HOG':
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            img = cv2.imread(img_path)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, feature_method, img_path))
            fd, hog_image = extract_hog(img)
            feature = fd
            ## Compute similatiy
            dist = compute_similarity(fd, features_storage, similarity_method )
            rank = np.argsort(dist)
            ranks.append(rank)
            
    elif feature_method == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, feature_method, img_path))
            img = cv2.imread(img_path)
            keypoints, des = extract_surf(img, surf)
            feature = des
            features.append(feature)

        
    elif feature_method == 'VGG16':
        model = VGG16(weights='imagenet', include_top=True)
        model.summary()
        print('input path', input_path)
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, feature_method, img_path))
            img = cv2.imread(img_path)
            feature = feature = extract_vgg16(img, model)
            features.append(feature)
    return ranks


def main(args):
    input_path = args['input_path']

    # Load feature
    file_name = args['feature_method'] + '.npz'
    if args['LSH'] == True:
        file_name = str(SIZE_PROJECTION) + '_LSH_'  + file_name
    feature_path = os.path.join(args['feature_path'],file_name)
    print('feature path: ', feature_path)
    data = np.load(feature_path)
    features_storage = data['features']
    path_storage = data['paths']
    print('Shape features: ', features_storage.shape)
    print('The path of img in storages: ', path_storage)

    # Start query
    ranks = retrieval_image(args["feature_method"], args["similarity_measure"], \
    input_path, features_storage )

    print("Ranks: ", ranks)

    # Show result

    if args["option"] == "query":
        pass
    elif args['option'] == "eval":
        pass
    else:
        print("[ERROR]: Pleas enter option again!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RETRIEVAL ONEPIECE IAMGE")
    parser.add_argument('-op', '--option',  default="query",
                        help="Choose option: \
                        if you want query, you will enter: query\
                        if you want eval you will enter: eval.")

    parser.add_argument('-i', '--input_path', default=".\data\eval",
                        help="The input path for query or eval")
    parser.add_argument('-o', '--output_path', default=".\data\eval",
                        help="The input path for query or eval")
    parser.add_argument('-fm', '--feature_method', default="SIFT",
                        help="Method to extrac feature. We can choose such as: SIFT, HOG, VGG16, SURF")
    parser.add_argument('-sm', '--similarity_measure', default="norm2",
                        help="Method similarity measure. We can choose such as: cosine, euclidean, manhatan, norm2")
    parser.add_argument('-fp', '--feature_path', default='./feature',
                        help="The feature path of storage")
    parser.add_argument('-lsh', '--LSH', default=False,
                        help="Use Locality-Sensitive Hasing ")

                      
    # End default optional arguments

    args = vars(parser.parse_args())

    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Option", args['option']).center(100))
    print("|{:<30}|{:<30}|".format("Input folder path", args['input_path']).center(100))
    print("|{:<30}|{:<30}|".format("Feature path", args['feature_path']).center(100))
    print("|{:<30}|{:<30}|".format("Feature method", args['feature_method']).center(100))
    print("|{:<30}|{:<30}|".format("Similarity measure", args['similarity_measure']).center(100))
    print("|{:<30}|{:<30}|".format("Use LSH", args['LSH']).center(100))

    print(str("-"*63).center(100))
    main(args)
