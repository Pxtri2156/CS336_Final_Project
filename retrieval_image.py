import cv2
import os

from tqdm import tqdm
from scipy.spatial import cKDTree
from skimage.measure import ransac

from similarity_measure import*
from extraction_method import *
from config import SIZE_PROJECTION, RANDOM_SEED, NUM_RESULT, DISTANCE_THRESOLD
from glob import glob

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
    
    elif feature_method == "DELF":
        print('input path', input_path)

        path_list = glob(input_path + "/*")
        print("path list ", path_list)
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.FATAL)

        m = hub.Module('https://tfhub.dev/google/delf/1')

        # The module operates on a single image at a time, so define a placeholder to
        # feed an arbitrary image in.
        image_placeholder = tf.placeholder(
            tf.float32, shape=(None, None, 3), name='input_image')

        module_inputs = {
            'image': image_placeholder,
            'score_threshold': 100.0,
            'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
            'max_feature_num': 1000,
        }

        module_outputs = m(module_inputs, as_dict=True)
        image_tf = image_input_fn(path_list)
        des_dic = extract_delf(image_tf, path_list, module_outputs, image_placeholder)
        querys_features = des_dic 
        
    
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




