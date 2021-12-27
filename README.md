# yolov3fromzero

tested on ubuntu1804 cuda11.4 NVIDIA RTX A5000

依赖包:
```
albumentations 1.0.2
torch 1.9.0
numpy 1.20.1
cv2 4.5.3
```

https://github.com/albumentations-team/albumentations/issues/459#issuecomment-734454278
数据增强库albumentations的bug需要参考上述链接手动修复.


使用前根据自己的代码的实际路径替换下面的`/home/autocore/work`
```
export PYTHONPATH=$PYTHONPATH:/home/autocore/work/yolov3fromzero:/home/autocore/work/yolov3fromzero/models
```

数据集存放于coco目录下.结构如下.
```
coco
├── images

│   ├── train2014
│   ├── train2017
│   ├── val2014
│   └── val2017
└── labels
    ├── train2014
    ├── train2017
    ├── val2014
    └── val2017
```

把该创建的目录创建好.

```
cd yolov3fromzero
mkdir input_imgs  //input_imgs目录用于存放原图经过处理后输入到模型的图片.
mkdir out_imgs    //out_imgs目录用于存放在原始图片上的检测效果图.
mkdir checkpoints //存放训练好的模型文件
mkdir log         //存放日志
```

## 测试数据处理
第一次训练前执行下面命令,以确保你的数据没有问题. 如果出错,请检查是否有不存在的图片或是label(包括缺失文件或者文件内容为空)
```
python dataset/dataset.py
```
会生成经过数据处理后送进模型的输入图片,位于input_imgs目录.


## 检测train.txt里是否有不存在label的图片
python coco/check_traintxt.py
如果有,修正traintxt. 新的文件命名为xxx.new
label的格式默认为yolo的格式. cxywh xywh为比例.


## 训练
python train.py



