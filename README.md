
# 车牌检测和识别

本项目使用基于YOLOv5和CNN的方式，实现了对车牌的检测和识别。
流程如下：

 1. 使用YOLOv5对数据集进行训练，得到车牌检测的模型
 2. 提取车牌并保存到新的文件夹中，训练提取后的车牌
 3. 设计卷积神经网络（CNN），完成车牌的识别

数据集文件夹分布(可到官网下载CCPD数据集)：

 ---dateset_DY
 
 ------Images
 
 ---------train
 
 -----------1.jpg
 
 -----------2.jpg
 
 ....
 
 ---------val
 
  -----------1.jpg
  
 -----------2.jpg
 
 ....
 
 ------labels
 
 ---------train
 
 ------------1.txt
 
 ------------2.txt
 
 ...
 
 ---------IMtrain_license_lables.csv
 
--
   *由于jpg和txt文件不是在同一文件夹中，需要把jpg和txt文件放到同一文件夹，才能进行yolo训练。*
   
 ## YOLOv5训练代码
 
```powershell
python train.py --data ../plate.yaml --cfg ./modesl/my_yolov5s.yaml --weights ./yolov5s.pt --batch-size 64 --workers 8 --epochs 100
```

# 程序内容
best.pt：训练好的yolov5模型
plateRec_model：车牌识别模型
plate_dec.py：根据yolov5提取出车牌的图片保存到train_process文件夹
train.ipynb：车牌识别训练文件
plate_rec.py：输入一张完整的图片可以识别出车牌
plate.yaml：yolo训练的yaml文件


# 输出结果
![输出结果](https://img-blog.csdnimg.cn/direct/7a64ac5f658a4e01ac5d5fb11d291d34.png)



