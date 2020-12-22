from extraction_method import*

def main(args):
    img = cv2.imread(args['input_path'])

    if args['method'] == 'sift':
        keypoints, des = extract_sift(img)
        print('Shape of decriptors: ', des.shape)
    elif args['method'] == 'hog':
        pass
    elif args['method'] == 'surf':
        pass
    elif args['method'] == 'vgg16':
        model = VGG16(weights='imagenet', include_top=True)
        model.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-i', '--input_folder',  default=".\data\train",
                        help="The path of the input image folder.")
    parser.add_argument('-o', '--output_folder', defalt=".\feature\SIFT",
                        help="The path of the output feature folder")
    parser.add_argument('-m', '--method', default="sift",
                        help="Method to extrac feature. We can choose such as: sift, hog, vgg16....")
    # End default optional arguments

    args = vars(parser.parse_args())

    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Image path", args['input_path']).center(100))
    print(str("-"*63).center(100))

    main(args)
