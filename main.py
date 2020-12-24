from similarity_measure import*
from extract_database import *

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
    parser.add_argument('-o', '--output_path', default=".\data\eval",
                        help="The input path for query or eval")
    parser.add_argument('-fm', '--feature_method', default="SIFT",
                        help="Method to extrac feature. We can choose such as: SIFT, HOG, VGG16, SURF")
    parser.add_argument('-sm', '--similarity_measure', default="norm2",
                        help="Method similarity measure. We can choose such as: cosine, euclidean, manhatan, norm2")
    parser.add_argument('-lsh', '--LSH', default=False,
                        help="Use Locality-Sensitive Hasing ")

                      
    # End default optional arguments

    args = vars(parser.parse_args())

    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Input folder path", args['input_path']).center(100))
    print("|{:<30}|{:<30}|".format("Option", args['option']).center(100))
    print("|{:<30}|{:<30}|".format("Feature method", args['feature_method']).center(100))
    print("|{:<30}|{:<30}|".format("Similarity measure", args['similarity_measure']).center(100))
    print("|{:<30}|{:<30}|".format("Use LSH", args['LSH']).center(100))

    print(str("-"*63).center(100))
