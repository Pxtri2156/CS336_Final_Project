import json
import numpy as np

def compute_AP(result,total):
  AP = np.sum( (np.arange(np.sum(result == "R")) + 1)/ (np.where(result == "R")[0] + 1)) / total
  return AP


def compute_mAP(ground_truth, ranks, paths, query_name):
    # Load ground truth 
    mAP = None
    ground_truth_file = open(ground_truth, 'r')
    data = json.load(ground_truth_file)
    # Take result from query result 
    print("type data: ", type(data))
    APs = []
    for i, rank in enumerate(ranks):
      
      query = query_name[i].split("/")[-1]
      print("Processing query {}".format(query))

      results = []
      for j, candidate in enumerate(rank):
          result = paths[candidate].split("/")[-1]
          # print("result: ", result)
          if result in data[query]:
            results.append("R")
          else:
            results.append("I")
  
      # print("Ground truth of query {}: \n{}".format(query, data[query]))
     
      results = np.array(results)
      total = len( data[query])
      AP = compute_AP(results,total)
      # print("AP of query {} : {}".format(query, AP))
      APs.append(AP)
    mAP = sum(APs) / ranks.shape[0]
    return mAP

def main(args):
    pass 

def args_parse():

    parser = argparse.ArgumentParser(description="Methods extract image.")
    parser.add_argument('-g', '--ground_truth',  default="/content/drive/MyDrive/Information_Retrieval/groundtruth/New_Ground_Truth/new_big_groundtruth.json",
                        help="The path of the input image folder.")

    return vars(parser.parse_args())

if __name__ == "__main__":
    
    args = args_parse()
    # Print info arguments
    print("Extract feature from image.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("groundtruth", args['ground_truth']).center(100))


    print(str("-"*63).center(100))

    main(args)