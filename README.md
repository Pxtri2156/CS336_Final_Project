# CS336_Final_Project
This is final project of information retrieval 

# My team 
 Name | Email| Git|
 --- | --- | ---
Phạm Xuân Trí | 18521530@gm.uit.edu.vn| [Pxtri2156](https://github.com/Pxtri2156)
Nguyễn Vương Thịnh | 18520367@gm.uit.edu.vn| [ducvuuit]()
Lưu Hoàng Sơn | 18521348@gm.uit.edu.vn| [ThinhNguyen209](https://github.com/ThinhNguyen209)
# Architecture code
## main.py
If you want to retrieval or eval, you will run this file
## similarity_mesure.py
This file give me many similarity measure: Cosine, Euclidean, Manhatan, Norm2
## feature_method.py
This file give me feature method to extract feature from image: SIFT, HOG, SURF, VGG16
## 
# Usage

## Exactract feature for storage
'''
python extract_database.py \\
--input_folder <input_path> \\
--output_folder  <out_path> \\
--method <feature_method> \\
--LSH 1
'''
**Example**: 
'''
python extract_database.py \\
--input_folder "/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/data/train" \\
--output_folder "/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/feature" \\
--method "HOG" \\
--LSH 1
'''
## Query image

!python main.py \\
--option="query" \\
--input_path=<input_path> \\
--output_path= <out_path> \\
--feature_path= <fearure_path_saved> \\
--feature_method=<extract feature method>\\
--similarity_measure=<compute_similarity_measure> \\
--LSH=1

**Example**:  
'''
!python main.py \\
--option="query" \\
--input_path="/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/data/test"  \\
--output_path="/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/data/valid" \\
--feature_path="/content/drive/MyDrive/Information_Retrieval/src/CS336_Final_Project/feature" \\
--feature_method="HOG" \\
--similarity_measure="cosine" \\
--LSH=1
'''





