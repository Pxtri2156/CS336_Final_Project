import cv2
import os

from tqdm import tqdm
from scipy.spatial import cKDTree
from skimage.measure import ransac

from similarity_measure.similarity_measure import *
from extraction.COLOR import COLOR
from extraction.DELF import DeepDELF
from extraction.SIFT import SIFT
from extraction.SURF import SURF
from extraction.VGG16 import DeepVGG16
from extraction.HOG import HOG 

from config import SIZE_PROJECTION, RANDOM_SEED, NUM_RESULT, DISTANCE_THRESOLD
from glob import glob



def retrieval_image(feature_method, similarity_method, input_path, features_storage,LSH,projections = None ):

    # Compute feature for all query 
    querys_features = []
    extractor = None
    if feature_method != "DELF":

        if feature_method == 'SIFT':
            pass
        elif feature_method == 'HOG':
            extractor = HOG()
        elif feature_method == 'SURF':
            pass
        elif feature_method == 'COLOR':
            extractor = COLOR()
        elif feature_method == 'VGG16':
            extractor = DeepVGG16()

        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("\n[INFO] Processing query: img: {}, f_method: {}, s_method: {}, LSH: {} \nquery_path: {}".format( \
            img_name, feature_method, similarity_method, LSH, img_path))
            img = cv2.imread(img_path)
            feature = extractor.extract(img)
            querys_features.append(feature)
    
    elif feature_method == "DELF":

        extractor = DeepDELF(input_path)
        des_dic = extractor.extract()
        path_list = list(des_dic.keys())
        querys_features = des_dic 
        print("Key feature: ", querys_features.keys())
        print("feature: ", type(querys_features))
          
    
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
    if feature_method != 'DELF':
      measure = None  
      if  similarity_method == "cosine":
          measure = Cosine_Measure()
      elif  similarity_method == "euclidean":
          measure =  Euclidean_Measure()
      elif similarity_method == "manhatan":
          measure = Manhatan_Measure()
      elif similarity_method == "lsh_IOU":
          measure = IOU_Measure(SIZE_PROJECTION)
      else:
        print('[ERROR]: Wrong similatity measure')

      for query in querys_features:
          print("[INFO]: Computing")
          print("shape feature storage: ", features_storage.shape )
          dist = measure.compute_similarity(query, features_storage ) 
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
    else:
      features_storage = features_storage[()] # Convert to dictionary 

      for query in querys_features.keys():
          locations_query, descriptors_query = querys_features[query]
          query_tree = cKDTree(descriptors_query)
          num_features_query = locations_query.shape[0]
          score = []

          for data in features_storage.keys():

              locations_data, descriptors_data = features_storage[data]
              num_features_data = locations_data.shape[0] 
              # Find nearest-neighbor matches using a KD tree.         
              _, indices = query_tree.query(
                  descriptors_data, distance_upper_bound=DISTANCE_THRESOLD)
              ##
              locations_data_to_use = np.array([locations_data[i,]
                  for i in range(num_features_data)
                  if indices[i] != num_features_query
              ])
              locations_query_to_use = np.array([
                  locations_query[indices[i],]
                  for i in range(num_features_data)
                  if indices[i] != num_features_query
              ])

            # Perform geometric verification using RANSAC.
              try:
                  _, inliers = ransac(
                      (locations_query_to_use, locations_data_to_use),
                      AffineTransform,
                      min_samples=3,
                      residual_threshold=20,
                      max_trials=1000)
                  inliers = sum(inliers)
                  ##
              except: 
                  inliers = 0
              # print('inliers: ', inliers)
              # print('key: ',data )
             
              
              score.append(inliers)
              # print('inliners', inliers)
              # print('score, ', score)
          score =  np.sort(np.array(score))[::-1]
          rank = np.argsort(np.array(score))[::-1]
      scores.append(score)
      ranks.append(rank)
    return ranks, scores

def main():
    pass

if __name__ == "__main__": 
    pass



