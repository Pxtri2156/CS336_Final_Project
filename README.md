# CS336_Final_Project
This is final project of information retrieval

# My team
|Name               | Email                 | Git                                                   |
|-------------------|-----------------------| ------------------------------------------------------|
|Phạm Xuân Trí      | 18521530@gm.uit.edu.vn| [Pxtri2156](https://github.com/Pxtri2156)             |
|Nguyễn Vương Thịnh | 18520367@gm.uit.edu.vn| [ThinhNguyen209](https://github.com/ThinhNguyen209)   |
|Lưu Hoàng Sơn      | 18521348@gm.uit.edu.vn| [sonlhcsuit](https://github.com/sonlhcsuit)           |


# Architecture code
## File main.py
If you want to query one or more image, you will run this file. And, you can run file to evaluation. 
## Folder extraction
Folder of feature extractions for image : *HOG, HSV-Histogram, VGG16, SURF, SIFT, DELF*
## Folder similarity_mesure
Folder of similarity measures : *Euclidean, Cosine, Manhatan, IOU*

# Usage

## Exactract feature for storage
To extract feature for dataset. You must run file **extract_database.py**.  
Argument:
* input_folder: The path of folder image that need extract feature.
* output_folder: The path of folder, where are save feature. 
* method: method feature extraction that you want to apply.
* LSH: Assign 1, if you want to activate LSH. 


```sh
python extract_database.py \\  
--input_folder <input_path> \\  
--output_folder  <out_path> \\  
--method <feature_method> \\  
--LSH 1  
```
**Example**:
```sh
python extract_database.py \\  
--input_folder "/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/data/train" \\  
--output_folder "/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/feature" \\  
--method "HOG" \\  
--LSH 1  
```

**NOTE**: If you are using colab, make sure you add '!' before the command.
## Query image
To query 1 or more image. You must run file **main.py**  
Argument: 
* option: query or eval.
* input_path: the path folder of query image.
* output_path: the path will save result query.
* feature_path: the path of feature dataset. 
* feature_method: method feature extraction that you want to apply.
* similarity_measure: method similarity measure that you want to apply.
* ground_truth: the path ground_truth. If you want to query, you can't enter this path.
```sh
python main.py \\  
--option="query" \\  
--input_path=<input_path> \\  
--output_path= <out_path> \\  
--feature_path= <fearure_path_saved> \\  
--feature_method=<extract feature method>\\  
--similarity_measure=<compute_similarity_measure> \\  
--LSH=1  \\
--ground_truth = None 
```
**Example**:  
```sh
python main.py \\  
--option="query" \\  
--input_path="/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/data/test"  \\  
--output_path="/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/data/valid" \\  
--feature_path="/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/feature" \\  
--feature_method="HOG" \\  
--similarity_measure="cosine" \\  
--LSH=1  \\
--ground_truth = None 
```
