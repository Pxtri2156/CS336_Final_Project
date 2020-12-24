import numpy as np
from tqdm import tqdm

from extraction_method import*
from config import SIZE_PROJECTION


def extract_database(input_path, method):
    features = [] # save feature
    path_list = [] # save path of each image
    if args['method'] == 'SIFT':
        sift =  cv2.xfeatures2d.SIFT_create()
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, method, img_path))
            img = cv2.imread(img_path)
            keypoints, des = extract_sift(img, sift)
            feature = des
            features.append(feature)
            path_list.append(img_path)
    elif args['method'] == 'HOG':
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            img = cv2.imread(img_path)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, method, img_path))
            fd, hog_image = extract_hog(img)
            feature = fd
            features.append(feature)
            path_list.append(img_path)
    elif args['method'] == 'SURF':
        surf = cv2.xfeatures2d.SURF_create()
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, method, img_path))
            img = cv2.imread(img_path)
            keypoints, des = extract_surf(img, surf)
            feature = des
            features.append(feature)
            path_list.append(img_path)

        
    elif args['method'] == 'VGG16':
        model = VGG16(weights='imagenet', include_top=True)
        model.summary()
        print('input path', input_path)
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path,img_name)
            print("[INFO] Processing: img: {} method: {} \npath_img: {}".format( \
            img_name, method, img_path))
            img = cv2.imread(img_path)
            feature = feature = extract_vgg16(img, model)
            features.append(feature)
            path_list.append(img_path)
    else:
       print("[ERROR]:Wrong method,  Pleas enter extract method again!!!")

    return np.array(features), path_list
        
def save_feature(features, path_list, output_path, method, LSH, k = SIZE_PROJECTION ):
  name_save_file = method + ".npz"
  if LSH == True:
    name_save_file = str(k) + '_LSH_' + name_save_file 

  save_path = os.path.join(output_path, name_save_file)
  np.savez_compressed(save_path, features=features, paths = path_list)

def main(args):

    # extract feature
    print("[INFO] Extracting  {} feature for dataset".format(args["method"]))
    features, path_list = extract_database(args['input_folder'], args['method'])
    print('len features',features.shape)
    # save feature 

    print("[INFO]: Begin save feature")
    save_feature(features,path_list,  args['output_folder'], args['method'], args['LSH'],SIZE_PROJECTION)
    print("[INFO]: Saved feature")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_folder',  default=".\data\train",
                        help="The path of the input image folder.")
    parser.add_argument('-o', '--output_folder', default=".\feature\SIFT",
                        help="The path of the output feature folder")
    parser.add_argument('-m', '--method', default="SIFT",
                        help="Method to extrac feature. We can choose such as: sift, hog, vgg16....")
    parser.add_argument('-lsh', '--LSH', default=False,
                        help="Use Locality-Sensitive Hasing ")
    # End default optional arguments

    args = vars(parser.parse_args())

    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Input path", args['input_folder']).center(100))
    print("|{:<30}|{:<30}|".format("Output path", args['output_folder']).center(100))
    print("|{:<30}|{:<30}|".format("Method path", args['method']).center(100))
    print("|{:<30}|{:<30}|".format("Use LSH", args['LSH']).center(100))

    print(str("-"*63).center(100))

    main(args)
