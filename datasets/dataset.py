from albumentations.augmentations.crops.functional import bbox_center_crop
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

class  LoadImagesAndLabels(Dataset):
    def __init__(self,traintxt,imgsize=416,debug=False,label_type='yolo',aug=False,mosaic=False):
        super().__init__()
        self.imgsize = imgsize
        try:
            with open(traintxt,'r') as f:
                lines = f.read().splitlines()
                # print(lines)
        except:
            raise Exception('{} does not exist'.format(traintxt))
        
        self.img_files = lines
        self.label_files = [x.replace('images', 'labels').replace(os.path.splitext(x)[-1], '.txt') for x in self.img_files] 
        # print(self.label_files)
        self.debug = debug
        self.label_type=label_type
        self.aug=aug
        self.mosaic = mosaic

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self,index):
        """
        return : new_label:[box_num,6] 6:img_idx_in_this_batch,c,x,y,w,h]
        """
        # print('__getitem__**********************')

        if self.mosaic:
            # 选取四张图片
            s = self.imgsize
            # xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
            indices = [index] + [random.randint(0, len(self.label_files) - 1) for _ in range(3)] 
            # print('选中的图片下标为:{}'.format(indices))
            img4_label4 = []
            for i, index in enumerate(indices):
                # load image
                img_path = self.img_files[index]
                # print('__getitem__:{}'.format(img_path))
                img = cv2.imread(img_path) 

                # load label
                label_path = self.label_files[index]
                try:
                    with open(label_path,'r') as f:
                        lines = f.read().splitlines()
                        label = [line.split() for line in lines]
                        label = np.array(label,dtype=np.float32)
                except:
                    raise Exception('{} does not exist'.format(label_path))

                img4_label4.append((img,label))    

            new_img,new_label = mosaic(self.imgsize,img4_label4,aug=self.aug)
        else:
            # load image
            img_path = self.img_files[index]
            # print('__getitem__:{}'.format(img_path))
            img = cv2.imread(img_path) 
            
            # load label
            label_path = self.label_files[index]
            try:
                with open(label_path,'r') as f:
                    lines = f.read().splitlines()
                    label = [line.split() for line in lines]
                    label = np.array(label,dtype=np.float32)
            except:
                raise Exception('{} does not exist'.format(label_path))
            
            if self.label_type == 'coco':
                    h,w = img.shape[0],img.shape[1]

                    box_lt_x = label[...,1]
                    box_lt_y = label[...,2]
                    box_rd_x = label[...,3]
                    box_rd_y = label[...,4]

                    box_center_x = (box_lt_x + box_rd_x)/2
                    box_center_y = (box_lt_y + box_rd_y)/2
                    box_w = (box_rd_x - box_lt_x)
                    box_h = (box_rd_y - box_lt_y)

                    label[...,1] = box_center_x/w
                    label[...,2] = box_center_y/h
                    label[...,3] = box_w/w
                    label[...,4] = box_h/h

                    # print(label)

            if self.aug:
                    transformed_image,transformed_bboxes,transformed_classes = augment_image(img,label)
                    img = transformed_image[:,:,::-1] #rgb-->bgr
                    label[...,0] = np.array(transformed_classes)
                    label[...,1:] = np.array(transformed_bboxes)
                    #对img的赋值不要放在augment_image里写. 注意python传参传引用的区别.

            # print('before letter_box,img shape:{},img_path:{}'.format(img.shape,img_path))
            new_img,new_label = letter_box(img,label,desired_size=(self.imgsize,self.imgsize))
            # print('after letter_box,img shape:{},img_path:{}'.format(img.shape,img_path))
            if new_img.shape[0] != 416 or new_img.shape[1] != 416:
                print('************************')

        if self.debug:
            debug_dataset(img_path,new_img,new_label)

        new_img = new_img[:,:,::-1] #bgr->rgb
        new_img = new_img.transpose( 2, 0, 1)  #hwc-chw
        new_img = np.ascontiguousarray(new_img)
        #https://www.cnblogs.com/devilmaycry812839668/p/13761613.html
        #返回的是chw rgb格式.
        # print(new_img.shape)

        return torch.from_numpy(new_img),torch.from_numpy(new_label),img_path

    @staticmethod
    def collate_fn(batch):
        # print('collate_fn**********************')
        img,label,img_path = list(zip(*batch))

        new_label=[]
        for i,l in enumerate(label):
            box_num,attr_num = l.shape
            new_l = torch.randn(box_num,1+attr_num)
            new_l[:,1:] = l
            new_l[:,0] = i
            #对label在dim1上添加1,标识这是哪一个图片的label

            new_label.append(new_l)

        imgs,labels = torch.stack(img,0),torch.cat(new_label,0)
        return imgs,labels,img_path

def letter_box(img,label,desired_size=(416,416),color=[114,114,114]):
    """
    把img resize到特定尺寸. 保持宽高比.
    desired_size : (h,w)

    return:尺寸为desired_size的img. label为目标在新的Img中的比例.
    """
    # print('img shape:{},label shape:{}'.format(img.shape,label.shape))

    origin_h,origin_w = img.shape[:2] # old_size is in (height, width) format
    desired_h,desired_w = desired_size[0],desired_size[1]
    # ratio = float(desired_size)/max(old_size)
    ratio_h = float(desired_h)/origin_h
    ratio_w = float(desired_w)/origin_w
    ratio = min(ratio_h,ratio_w)
    # print('ratio={},desired_h:{},desired_w:{}'.format(ratio,desired_h,desired_w))
    new_size = tuple([round(x*ratio) for x in (origin_h,origin_w)])
    
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    #https://blog.csdn.net/guyuealian/article/details/85097633 如何选择插值的方式
    img = cv2.resize(img,(new_size[1],new_size[0]),interpolation=cv2.INTER_AREA)
    # 这里只改变了img的尺寸.宽高比是没有变化的

    # new_size should be in (width, height) format
    # im = cv2.resize(img, (new_size[1], new_size[0]))
    # print('old_size:{} new_size:{}'.format((origin_h,origin_w),new_size))
    # print('img shape:{}'.format(img.shape))

    h,w = img.shape[:2]
    box_x = w * label[...,1]
    box_y = h * label[...,2]
    box_w = w * label[...,3]
    box_h = h * label[...,4]
    # print(box_x.shape,box_y.shape,box_w.shape,box_h.shape)
    # 这里是绝对尺度.不是比例

    delta_w = max(0,desired_size[1] - new_size[1])
    delta_h = max(0,desired_size[0] - new_size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    # print('top:{},bottom:{},left:{},right:{}'.format(top,bottom,left,right))
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    # print('new_img shape:{}'.format(new_img.shape))
    new_img_h,new_img_w = new_img.shape[:2]
    if new_img_h != desired_size[0]:
        print('img:{},top:{},bottom:{},left:{},right:{}'.format(img.shape,top,bottom,left,right))
    new_label = label    
    # c x y w h
    new_label[:,1] = (left  + box_x)/new_img_w
    new_label[:,2] = (top  + box_y)/new_img_h
    new_label[:,3] = box_w/new_img_w
    new_label[:,4] = box_h/new_img_h

    # print('after letter_box img shape:{}'.format(new_img.shape))
    return new_img,new_label

import albumentations as A
import random
def augment_image(img,label):
    """
    img,label:ndarray
    albumentations用的是rgb顺序
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #albumentations用的是rgb顺序
    
    transform = A.Compose([
        # A.RandomCrop(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        # A.RandomRotate90(),
        # A.Rotate(limit=25, p=0.2),  # 限制旋转角度为25度
        A.GaussNoise(p=0.2),
        A.GlassBlur(p=0.2),
        A.RandomGamma(p=0.2),
        # A.RandomRain(p=0.1),
        # A.RandomSunFlare(p=0.1),
        # A.CenterCrop(height=50,width=50) #从图像中间裁剪出h*w的区域
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    
    box_num = label.shape[0]
    bboxes = [[]]*box_num
    category_ids = []

    # print(label)
    for i in range(box_num):
        cls,x,y,w,h =label[i,0],label[i,1],label[i,2],label[i,3],label[i,4]
        bboxes[i] = [x,y,w,h]
        category_ids.append(cls)
    # print('bboxes:{}'.format(bboxes))

    random.seed()
    transformed = transform(image=img_rgb, bboxes=bboxes, category_ids=category_ids)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_classes = transformed['category_ids']

    # print(label)
    return transformed_image,transformed_bboxes,transformed_classes

def mosaic(img_size,img4_label4,aug=False):
    """
    将多张图片拼接在一起.
    img_size:拼接后的图像大小.
    img1/2/3/4:待拼接的四张图
    label1/2/3/4:yolo格式的标签. cxywh xywh代表比例.
    """
    new_img = np.zeros((img_size,img_size,3),dtype=np.uint8)
    new_label = []
    divid_point_x, divid_point_y = [int(random.uniform(img_size * 0.3,img_size * 0.7)) for _ in range(2)] 
    #生成一个随机的分割点

    for i in range(len(img4_label4)):
        img,label = img4_label4[i][0],img4_label4[i][1]

        if aug:
            transformed_image,transformed_bboxes,transformed_classes = augment_image(img,label)
            img = transformed_image[:,:,::-1] #rgb-->bgr
            label[...,0] = np.array(transformed_classes)
            label[...,1:] = np.array(transformed_bboxes) 

            img4_label4[i] = (img,label)
        
        img,label = img4_label4[i][0],img4_label4[i][1]
        if i == 0:
            #top-left
            img_w,img_h = divid_point_x , divid_point_y
            offset_x,offset_y = 0,0
            #offset代表这个区域的左上角的点位于拼接图中的坐标
        elif i == 1:
            #top-right
            img_w,img_h = img_size - divid_point_x , divid_point_y
            offset_x,offset_y = divid_point_x,0
        elif i == 2:
            #bottom-left
            img_w,img_h = divid_point_x , img_size - divid_point_y
            offset_x,offset_y = 0,divid_point_y
        elif i == 3:
            #bottom-right
            img_w,img_h = img_size - divid_point_x , img_size - divid_point_y
            offset_x,offset_y = divid_point_x,divid_point_y
        else:
            pass
        
        # img = cv2.resize(img, (img_w,img_h))
        img,label = letter_box(img,label,desired_size=(img_h,img_w),color=[114,114,114])
        img_h_o,img_w_o = img.shape[0],img.shape[1]
        box_num = label.shape[0]
        for j in range(box_num):
            cls,x,y,w,h =label[j,0],label[j,1],label[j,2],label[j,3],label[j,4]
            
            bbox_center_x = offset_x + img_w * x 
            bbox_center_y = offset_y + img_h * y
            bbox_w = img_w_o * w 
            bbox_h = img_h_o * h                 
            #box在整个拼接图中的中心点和宽高

            new_x = bbox_center_x/img_size
            new_y = bbox_center_y/img_size       
            new_w = bbox_w/img_size
            new_h = bbox_h/img_size
            #在拼接图中的比例

            new_label.append([cls,new_x,new_y,new_w,new_h])
        # scale_x,scale_y = img_w/img_w_o,img_h/img_h_o
        # box_num = label.shape[0]
        # for j in range(box_num):
        #     cls,x,y,w,h =label[j,0],label[j,1],label[j,2],label[j,3],label[j,4]

        #     bbox_center_x = offset_x + img_w_o * x *scale_x
        #     bbox_center_y = offset_y + img_h_o * y * scale_y
        #     bbox_w = img_w_o * w *scale_x
        #     bbox_h = img_h_o * h *scale_y                
        #     #resize以后的Box的中心点和宽高

        #     new_x = bbox_center_x/img_size
        #     new_y = bbox_center_y/img_size       
        #     new_w = bbox_w/img_size
        #     new_h = bbox_h/img_size
        #     #在拼接图中的比例

        #     new_label.append([cls,new_x,new_y,new_w,new_h])

        new_img[offset_y:offset_y+img_h,offset_x:offset_x+img_w,:] = img

    return new_img,np.asarray(new_label)

def debug_dataset(path,new_img,new_label):
    """
    debug处理后的图像和label是否正确
    path:原始图像路径
    new_img:处理后的img  hwc bgr
    new_label:处理后的img上的label
    """
    # print(new_img.shape,new_label.shape)
    # print('new_label={}'.format(new_label))
    name = path.split('/')[-1]
    full_name = './input_imgs/{}'.format(name)
    
    img_h,img_w,c  = new_img.shape
    x = new_label[:,1]
    y = new_label[:,2]
    w = new_label[:,3]
    h = new_label[:,4]
    cls = new_label[:,0]

    # print(x,y,w,h)

    lt_x = img_w * x - (img_w * w)/2
    lt_y = img_h * y - (img_h * h)/2
    rd_x = img_w * x + (img_w * w)/2
    rd_y = img_h * y + (img_h * h)/2

    # print('lt_x:{},lt_y:{},rd_x:{},rd_y:{}'.format(lt_x,lt_y,rd_x,rd_y))

    box_num = new_label.shape[0]
    tl = round(0.002 * (new_img.shape[0] + new_img.shape[1]) / 2) + 1  # line/font thickness
    for i in range(box_num):
        c1 = (int(lt_x[i]),int(lt_y[i]))
        c2 = (int(rd_x[i]),int(rd_y[i]))
        # print('c1:{},c2:{}'.format(c1,c2))
        tf = max(tl - 1, 1)  # font thickness
        cv2.rectangle(new_img, c1, c2, (255,0,0))
        cls_id=int(cls[i])
        cls_name = cls_names[cls_id]
        cv2.putText(new_img, cls_name, (c1[0], c1[1] - 2), 0, tl / 8, [225, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(full_name,new_img)                                    
    print('save {}'.format(full_name))

import datetime
if __name__ == '__main__':
    from utils.utils import *
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cls_names_path', type=str,default='coco/names', help='class names file')
    opt = parser.parse_args()
    cls_names = load_classes(opt.cls_names_path)

    # traintxt = '/home/autocore/work/yolov3_darknet/data/lishui/train.txt'
    root_dir=os.getcwd()
    traintxt = 'coco/val2017.txt'
    # traintxt = 'coco/train2017.txt'
    traintxt = root_dir + '/' + traintxt
    dataset = LoadImagesAndLabels(traintxt,debug=True,label_type='yolo',aug=False,mosaic=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        num_workers=1,
                                        shuffle=False,
                                        collate_fn=dataset.collate_fn
                                        )
    start=datetime.datetime.now()
    for i,data in enumerate(dataloader):
        img,label,path = data
        # print(path)
        
        # if i > 10:
        #     break

    end=datetime.datetime.now()
    print('耗时{}'.format(end - start))
        # break
    # img = cv2.imread('/home/autocore/work/yolov3_darknet/lishui/images/1599039225trans4.png')
    # newimg = letter_box(img,desired_size=416)
    # print(newimg.shape)