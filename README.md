# MobileNetV3-SSD
MobileNetV3-SSD implementation in PyTorch 

**目的**
Object Detection 
应用于目标检测

环境 

操作系统: Ubuntu18.04

Python: 3.6

PyTorch: 1.1.0


**使用MobileNetV3-SSD实现目标检测**

**Support Export ONNX**

代码参考（严重参考以下代码）


**一 SSD部分**


[A PyTorch Implementation of Single Shot MultiBox Detector ](https://github.com/amdegroot/ssd.pytorch)

**二 MobileNetV3 部分**



[1 mobilenetv3 with pytorch，provide pre-train model](https://github.com/xiaolai-sqlai/mobilenetv3) 


[2 MobileNetV3 in pytorch and ImageNet pretrained models ](https://github.com/kuan-wang/pytorch-mobilenet-v3)


[3Implementing Searching for MobileNetV3 paper using Pytorch ](https://github.com/leaderj1001/MobileNetV3-Pytorch)


[4 MobileNetV1, MobileNetV2, VGG based SSD/SSD-lite implementation in Pytorch 1.0 / Pytorch 0.4. Out-of-box support for retraining on Open Images dataset. ONNX and Caffe2 support. Experiment Ideas like CoordConv. 
no discernible latency cost](https://github.com/qfgaohao/pytorch-ssd).


针对4我这里没有做MobileNetV1, MobileNetV2等等代码兼容，只有MobileNetV3可用

**下载数据**
本例是以蛋糕和面包为例，原因是数据量小
所有类别总大小是561G，蛋糕和面包是3.2G

python3 open_images_downloader.py --root data/open_images --class_names "Cake,Bread" --num_workers 20


**训练过程**

**首次训练**

4GPU-/media/bizon/DATA/Documents/Study/mobileNetV3/MobileNetV3-Pytorch/README.md

**Open Images Dataset**
python3 train_ssd.py --dataset_type open_images --datasets data/open_images --net mb3-ssd-lite  --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 100 --base_net_lr 0.001  --batch_size 128

**Pascal VOC**
rm models/mb*; 
python3 train_ssd.py --dataset_type voc --datasets /media/bizon/DATA/Documents/Study/dataSet/voc/VOC2007 --net mb3-ssd-lite  --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 200 --base_net_lr 0.001  --batch_size 128

***VideoProcessing/trainingData***
rm models/mb*;
python3 train_ssd.py --dataset_type voc --datasets /media/bizon/DATA/Documents/Study/WHR_GitHub/VideoProcessing/trainingData --net mb3-ssd-lite  --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 2000 --base_net_lr 0.001  --batch_size 64



**预加载之前训练的模型**

python3 train_ssd.py --dataset_type open_images --datasets data/open_images --net mb3-ssd-lite --pretrained_ssd models/mb3-ssd-lite-Epoch-99-Loss-3.0880880981548877.pth  --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 200 --base_net_lr 0.001  --batch_size 5

python3 train_ssd.py --dataset_type voc --datasets /media/bizon/DATA/Documents/Study/dataSet/voc/VOC2007 --net mb3-ssd-lite --pretrained_ssd models/mb3-ssd-lite-Epoch-999-Loss-4.7395658242075065.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs 2000 --base_net_lr 0.001  --batch_size 128


**测试一张图片**

**Open Images Dataset**
python3 run_ssd_example.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-99-Loss-3.0880880981548877.pth models/open-images-model-labels.txt data/open_images/test/1d0adfc5f12e7ac0.jpg
**Pascal VOC**
python3 run_ssd_example.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-99-Loss-6.076028108596802.pth models/voc-model-labels.txt /media/bizon/DATA/Documents/Study/dataSet/voc/VOC2007/JPEGImages/000001.jpg
***VideoProcessing/trainingData***
python3 run_ssd_example.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-1999-Loss-2.280820846557617.pth models/voc-model-labels.txt /media/bizon/DATA/Documents/Study/WHR_GitHub/VideoProcessing/trainingData/JPEGImages/bicycle_right_644.jpg 


**视频检测**

python3 run_ssd_live_demo.py mb3-ssd-lite models/mb3-ssd-lite-Epoch-99-Loss-2.5194434596402613.pth models/open-images-model-labels.txt 

***VideoProcessing/trainingData***
python3 run_ssd_live_demo.py mb3-ssd-lite Trained_model/mb3-ssd-lite-Epoch-1999-Loss-2.4244446754455566-2020_02_05.pth models/voc-model-labels.txt /media/bizon/DATA/Documents/Study/WHR_GitHub/VideoProcessing/roadVideo_training/Supplementary_Movie_1_full-1239-1305.mp4

python3 processingVideo.py mb3-ssd-lite Trained_model/mb3-ssd-lite-Epoch-1999-Loss-2.4244446754455566-2020_02_05.pth models/voc-model-labels.txt /media/bizon/DATA/Documents/Study/WHR_GitHub/VideoProcessing/roadVideo_training/Supplementary_Movie_1_full-1239-1305.mp4


**Cake and Bread Pretrained model**


链接: https://pan.baidu.com/s/1byY1eJk3Hm3CTp-29KirxA 

提取码: qxwv 

**VOC Dataset Pretrained model**

链接: https://pan.baidu.com/s/1yt_IRY0RcgSxB-YwywoHuA 

提取码: 2sta 
