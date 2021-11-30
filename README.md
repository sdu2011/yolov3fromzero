# yolov3fromzero

export PYTHONPATH=$PYTHONPATH:/home/autocore/work/yolov3fromzero:/home/autocore/work/yolov3fromzero/models

input_imgs目录用于存放原图经过处理后输入到模型的图片.
out_imgs目录用于存放在原始图片上的检测效果图.

```
cd yolov3fromzero
mkdir input_imgs
mkdir out_imgs
```

## 检测train.txt里是否有不存在label的图片
python coco/check_traintxt.py
如果有,修正traintxt. 新的文件命名为xxx.new

## 测试数据处理
python dataset/dataset.py

## 训练
python train.py