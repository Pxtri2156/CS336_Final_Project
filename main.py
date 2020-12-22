def main(args):
    if args["option"] == "query":
        pass
    elif args['option'] == "eval":
        pass
    else:
        print("[ERROR]: Pleas enter option again!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RETRIEVAL ONEPIECE IAMGE")
    parser.add_argument('-op', '--option',  required="eval",
                        help="Choose option: \\
                        if you want query, you will enter: query\\
                        if you want eval you will enter: eval.")

    parser.add_argument('-i', '--input_path', default=".\data\eval",
                        help="The input path for query or eval")
    parser.add_argument('-m', '--method', default="sift",
                        help="Method to extrac feature. We can choose such as: sift, hog, vgg16....")
    # End default optional arguments

    args = vars(parser.parse_args())

    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Image path", args['input_path']).center(100))
    print(str("-"*63).center(100))
