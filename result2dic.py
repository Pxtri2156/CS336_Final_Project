import argparse
import numpy as np
from random import sample
import json
def result2csv(input_path, out_path):
    data = np.load(input_path)
    ranks = data['ranks']
    paths = data['paths']
    querys_name = data['query_name']
    results = {}
    
    for i, query in enumerate(querys_name):
        query_name = query.split("/")[-1]
        result_file = []
        for rank in ranks[i]:
            result_file.append(paths[rank].split("/")[-1])
        result_file = sample(result_file, 20)
        results[query_name] = result_file 

    fl = open(out_path, 'w')
    data = json.dumps(results,indent = 4)
    print('dictionary: ', data)

    json.dump(data, fl)


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
