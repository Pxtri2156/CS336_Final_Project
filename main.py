import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from skimage.measure import ransac
from retrieval_image import retrieval_image
from config import SIZE_PROJECTION, RANDOM_SEED, NUM_RESULT, DISTANCE_THRESOLD
from glob import glob


def save_result(rank,path_storage, score, query_name, save_file):
    
    np.savez_compressed(save_file, ranks = rank, paths = path_storage, scores = score, query_name = query_name)
        
def main(args):

    input_path = args['input_path']
    # Load feature
    print("[STATUS]:================Loading feature from storage ==================")
    ## Create name file 
    file_name = args['feature_method'] + '.npz'
    
    if args['LSH'] == True:
        file_name = str(SIZE_PROJECTION) + '_LSH_'  + file_name
    feature_path = os.path.join(args['feature_path'],file_name)
    print('feature path: ', feature_path)
    ## Loading file 
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
    query_name = glob(args['input_path'] + '/*')
    save_result( ranks[:, :NUM_RESULT], path_storage, scores,query_name, result_path)
    print("[INFO]: Saved result file")
    if args["option"] == "query":
        pass
    elif args['option'] == "eval":
        pass
    else:
        print("[ERROR]: Pleas enter option again!!!")

def args_parse():

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

    return vars(parser.parse_args())

if __name__ == "__main__":

    args = args_parse()
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
