# CS336_Final_Project
This is final project of information retrieval

# My team
|Name               | Email                 | Git                                                   |
|-------------------|-----------------------| ------------------------------------------------------|
|Phạm Xuân Trí      | 18521530@gm.uit.edu.vn| [Pxtri2156](https://github.com/Pxtri2156)             |
|Nguyễn Vương Thịnh | 18520367@gm.uit.edu.vn| [ThinhNguyen209](https://github.com/ThinhNguyen209)   |
|Lưu Hoàng Sơn      | 18521348@gm.uit.edu.vn| [sonlhcsuit](https://github.com/sonlhcsuit)           |


# Architecture code
## main.py
File này cho phép truy vấn một hoặc nhiều ảnh bất kì. Nó cũng là file chạy để đánh giá kết quả truy vấn.
## Folder extraction
Chứa các phương pháp rút trích đặc trưng cho ảnh: HOG, HSV-Histogram, VGG16, SURF, SIFT, DELF
## Folder similarity_mesure
Chứa cái độ đo tương đồng giữa 2 vector: Euclidean, Cosine, Manhatan, IOU

# Usage

## Exactract feature for storage
Để rút trích đặc trung cho tập dataset ( tập chúng ta sẽ truy vấn). Chúng ta cần run file **extract_database.py**.  
Các tham số môi trường khi chạy file bao gồm:
* input_folder: Đường dẫn của folder chứa ảnh cần rút trích đặc trưng.
* output_folder: Thư mục lưu trữ các vector đặc trưng sau khi rút trích các đặc trưng.
* method: phương


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

```sh
python main.py \\  
--option="query" \\  
--input_path=<input_path> \\  
--output_path= <out_path> \\  
--feature_path= <fearure_path_saved> \\  
--feature_method=<extract feature method>\\  
--similarity_measure=<compute_similarity_measure> \\  
--LSH=1  
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
--LSH=1  
```
