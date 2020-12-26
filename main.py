import cv2
import os

from tqdm import tqdm

from similarity_measure import*
from extraction_method import *
from config import SIZE_PROJECTION, RANDOM_SEED, NUM_RESULT

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
def retrieval_image(feature_method, similarity_method, input_path, features_storage,LSH,projections = None ):

    # Compute feature for all query 
    querys_features = []
    if feature_method == 'SIFT':
        sift =  cv2.xfeatures2d.SIFT_create()
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("\n[INFO] Processing query: img: {}, f_method: {}, s_method: {}, LSH: {} \nquery_path: {}".format( \
            img_name, feature_method, similarity_method, LSH, img_path))
            img = cv2.imread(img_path)
            keypoints, des = extract_sift(img, sift)
            feature = des
            querys_features.append(feature)

    elif feature_method == 'HOG':
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            img = cv2.imread(img_path)
            print("\n[INFO] Processing query: img: {}, f_method: {}, s_method: {}, LSH: {} \nquery_path: {}".format( \
            img_name, feature_method, similarity_method, LSH, img_path))
            fd, hog_image = extract_hog(img)
            feature = fd
            ## Compute similatiy
            querys_features.append(feature)
            
    elif feature_method == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("\n[INFO] Processing query: img: {}, f_method: {}, s_method: {}, LSH: {} \nquery_path: {}".format( \
            img_name, feature_method, similarity_method, LSH, img_path))
            img = cv2.imread(img_path)
            keypoints, des = extract_surf(img, surf)
            feature = des
            querys_features.append(feature)
    elif feature_method == 'COLOR':
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            img = cv2.imread(img_path)
            print("\n[INFO] Processing query: img: {}, f_method: {}, s_method: {}, LSH: {} \nquery_path: {}".format( \
            img_name, feature_method, similarity_method, LSH, img_path))
            hsv_hist = extrac_histogram(img)
            feature = hsv_hist
            ## Compute similatiy
            querys_features.append(feature)
    elif feature_method == 'VGG16':
        model = VGG16(weights='imagenet', include_top=True)
        model.summary()
        print('input path', input_path)
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("\n[INFO] Processing query: img: {}, f_method: {}, s_method: {}, LSH: {} \nquery_path: {}".format( \
            img_name, feature_method, similarity_method, LSH, img_path))
            img = cv2.imread(img_path)
            feature = feature = extract_vgg16(img, model)
            querys_features.append(feature)
    
    # Compute similarity 
    ranks = []
    scores = []
    if LSH == 1:
      if similarity_method != 'lsh_IOU':
        print("[ERROR]: With activate LSH, you must choose similarty measure is lsh_IOU")
        return None
      # Hasing query image
      print("[INFO]: Hasing query")
      querys_features = np.apply_along_axis(signature_bit,1,querys_features,projections)

    print("[STATUS]:================Compute similarity==================")      
    for query in querys_features:
        print("[INFO]: Computing")
        print("shape feature storage: ", features_storage.shape )
        dist = compute_similarity(query, features_storage, similarity_method ) 
        print('Shape dist: ', dist.shape)
        if similarity_method != 'cosine':
          score = np.sort(dist)
          rank = np.argsort(dist)
        else: 
          score = np.sort(dist)[::-1]
          rank = np.argsort(dist)[::-1]
        ranks.append(rank)
        scores.append(score)
        print("Top 10 score: ", score[:10])
        print("Top 10 rank: ", rank[:10])
    return ranks, scores


def save_result(rank,path_storage, score, save_file):
    
    np.savez_compressed(save_file, ranks = rank, paths = path_storage, scores = score)
        
def main(args):
    input_path = args['input_path']

    # Load feature
    print("[STATUS]:================Loading feature from storage ==================")
    file_name = args['feature_method'] + '.npz'
    
    if args['LSH'] == True:
        file_name = str(SIZE_PROJECTION) + '_LSH_'  + file_name
    feature_path = os.path.join(args['feature_path'],file_name)
    print('feature path: ', feature_path)

    data = np.load(feature_path,  allow_pickle=True)
    features_storage = data['features']
    path_storage = data['paths']
    print('Shape features: ', features_storage.shape)
    # print('The path of img in storages: ', path_storage)

    ## Load projections if LSH: True
    print("[STATUS]:================Loading projections ==================")
    projections = None
    if args['LSH'] == True: 
        projections = data['projections']
        print("[INFO]: Loaded projections")
    
    # Start query

    print("[STATUS]:================Retrieving with query images ==================")
    ranks, scores = retrieval_image(args["feature_method"], args["similarity_measure"], \
    input_path, features_storage,args["LSH"] , projections )

    # Show result
    if ranks == None:
        print("No result. Maybe, you run error some where !")
    else:
        ranks = np.array(ranks)
        print('Shape ranks: ',ranks.shape)
        print("Ranks: ", ranks[:, :NUM_RESULT].shape)
    
    # Save result
    print("[STATUS]:================  Wrting result ==================")
    result_file = str(NUM_RESULT) + "_" +  args["feature_method"] + '_' + args["similarity_measure"] + '_' + str(args["LSH"]) + '.npz'
    result_path = os.path.join(args["output_path"], result_file)
    print("Result path: ", result_path)
    save_result( ranks[:, :NUM_RESULT], path_storage, scores, result_path)
    print("[INFO]: Saved result file")
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
    parser.add_argument('-o', '--output_path', default=".\data\result",
                        help="The input path for query or eval")
    parser.add_argument('-fm', '--feature_method', default="SIFT",
                        help="Method to extrac feature. We can choose such as: SIFT, HOG, VGG16, SURF")
    parser.add_argument('-sm', '--similarity_measure', default="norm2",
                        help="Method similarity measure. We can choose such as: cosine, euclidean, manhatan, norm2")
    parser.add_argument('-fp', '--feature_path', default='./feature',
                        help="The feature path of storage")
    parser.add_argument('-lsh', '--LSH', default= 0,
                        help="Use Locality-Sensitive Hasing ", type = int)

                      
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
