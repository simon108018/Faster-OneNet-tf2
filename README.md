Faster OneNet

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [评估步骤 How2eval](#评估步骤)
8. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 |                                                           权值文件名称                                                            | 测试数据集 | 输入图片大小  | mAP 0.5:0.95 | mAP 0.5 |
| :-----: |:---------------------------------------------------------------------------------------------------------------------------:| :------: |:-------:|:------------:| :-----: |
| VOC07+12 | [faster_onenet_resnet18.h5](https://github.com/simon108018/Faster-OneNet/releases/download/v1.0/faster_onenet_resnet18.h5)  | VOC-Test07 | 320x320 |      -       | -
| VOC07+12 | [faster_onenet_resnet50.h5](https://github.com/simon108018/Faster-OneNet/releases/download/v1.0/faster_onenet_resnet18.h5)                                                | VOC-Test07 | 320x320 |      -       | -

## 所需环境
tensorflow-gpu==2.2.0  
由于tensorflow2中已经有keras部分，所以不需要额外装keras

## 注意事项
代码中的faster_onenet_resnet18.h5是使用voc数据集训练的。    
代码中的faster_onenet_resnet50.h5是使用voc数据集训练的。   
**注意不要使用中文标签，文件夹中不要有空格！**     
**在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**。     

## 文件下载 
训练所需的faster_onenet_resnet18.h5、faster_onenet_resnet50.h5可在上方下載。 

faster_onenet_resnet18.h5是voc数据集的权重。    
faster_onenet_resnet50.h5是coco数据集的权重。    

請先下載VOC Datasets

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载faster_onenet_resnet18.h5或者faster_onenet_resnet50.h5，放入model_data，运行predict.py，输入  
```python
img/street.jpg
```
2. 利用video.py可进行摄像头检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/faster_onenet_resnet18.h5',
    "classes_path"      : 'model_data/voc_classes.txt',
    #--------------------------------------------------------------------------#
    #   用于选择所使用的模型的主干
    #   resnet18, resnet50
    #--------------------------------------------------------------------------#
    "backbone"          : 'resnet18',
    #--------------------------------------------------------------------------#
    #   输入图片的大小
    #--------------------------------------------------------------------------#
    "input_shape"       : [320, 320],
    #--------------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #--------------------------------------------------------------------------#
    "confidence"        : 0.2,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    #--------------------------------------------------------------------------#
    #   是否进行非极大抑制，可以根据检测效果自行选择
    #   backbone为resnet50时建议设置为True、backbone为hourglass时建议设置为False
    #--------------------------------------------------------------------------#
    "nms"               : True,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
}

```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 利用video.py可进行摄像头检测。  

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在训练前利用voc2centernet.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8. 运行train.py即可开始训练。

## 评估步骤 
### a、评估VOC07+12的测试集
1. 本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在centernet.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

### b、评估自己的数据集
1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在centernet.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/bubbliiiing/centernet-tf2
https://github.com/PeizeSun/OneNet
https://github.com/xuannianz/keras-CenterNet      
https://github.com/see--/keras-centernet      
https://github.com/xingyizhou/CenterNet 
