import argparse
import numpy as np
from random import sample
import json
def result2csv(input_path, out_path):
    data = np.load(input_path)
    ranks = data['ranks']
    paths = data['paths']
    results = {}
    
    for i, rank in enumerate(ranks):

        result = []
        for j, candidate in enumerate(rank):
          path = paths[rank][j].split("/")[3]
          result.append(path)
        # print('result: ', result[:20])
        result = sample(result,20)
        results[i] = result

    # print('results: ',results)
    # print('output: ', out_path)
    print("out path: ", out_path)
    print("results: ", type(results))
    print("len dictionary: ", len(results))
    fl = open(out_path, 'w')
    json.dump(results, fl)


def main(args):

    result2csv(args['input_path'], args['output_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RETRIEVAL ONEPIECE IAMGE")
    parser.add_argument('-i', '--input_path',  default="None",
                        help="The path ressult.")

    parser.add_argument('-o', '--output_path', default="None",
                        help="The  path output of csv")

    args = vars(parser.parse_args())
    main(args)
