from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

class  LoadImagesAndLabels(Dataset):
    def __init__(self,traintxt,imgsize=416):
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
        print(self.label_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self,index):
        """
        return : new_label:[box_num,6] 6:img_idx_in_this_batch,c,x,y,w,h]
        """
        # print('__getitem__**********************')

        # load image
        img_path = self.img_files[index]
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
        
        new_img,new_label = letter_box(img,label,desired_size=416)

        # debug_dataset(img_path,new_img,new_label)

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

def letter_box(img,label,desired_size=416,color=[114,114,114]):
    # print('img shape:{},label shape:{}'.format(img.shape,label.shape))

    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([1+int(x*ratio) for x in old_size])
    if ratio != 1:
        interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        #https://blog.csdn.net/guyuealian/article/details/85097633 如何选择插值的方式
        img = cv2.resize(img,(new_size[1],new_size[0]),interpolation=cv2.INTER_AREA)
        # 这里只改变了img的尺寸.宽高比是没有变化的

    # new_size should be in (width, height) format
    # im = cv2.resize(img, (new_size[1], new_size[0]))
    # print('old_size:{} new_size:{}'.format(old_size,new_size))
    # print('img shape:{}'.format(img.shape))

    h,w = img.shape[:2]
    box_x = w * label[...,1]
    box_y = h * label[...,2]
    box_w = w * label[...,3]
    box_h = h * label[...,4]
    # print(box_x.shape,box_y.shape,box_w.shape,box_h.shape)
    # 这里是绝对尺度.不是比例

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    # print('new_img shape:{}'.format(new_img.shape))
    new_img_h,new_img_w = new_img.shape[:2]

    new_label = label    
    # c x y w h
    new_label[:,1] = (left  + box_x)/new_img_w
    new_label[:,2] = (top  + box_y)/new_img_h
    new_label[:,3] = box_w/new_img_w
    new_label[:,4] = box_h/new_img_h

    return new_img,new_label

def debug_dataset(path,new_img,new_label):
    """
    debug处理后的图像和label是否正确
    path:原始图像路径
    new_img:处理后的img  hwc bgr
    new_label:处理后的img上的label
    """
    print(new_img.shape,new_label.shape)
    name = path.split('/')[-1]
    full_name = './input_imgs/{}'.format(name)
    
    img_h,img_w,c  = new_img.shape
    x = new_label[:,1]
    y = new_label[:,2]
    w = new_label[:,3]
    h = new_label[:,4]

    lt_x = img_w * x - (img_w * w)/2
    lt_y = img_h * y - (img_h * h)/2
    rd_x = img_w * x + (img_w * w)/2
    rd_y = img_h * y + (img_h * h)/2

    cv2.rectangle(new_img, (lt_x,lt_y), (rd_x,rd_y), (255,0,0))
    cv2.imwrite(full_name,new_img)
    print('save {}'.format(full_name))

if __name__ == '__main__':
    traintxt = '/home/autocore/work/yolov3_darknet/data/lishui/train.txt'
    dataset = LoadImagesAndLabels(traintxt)
    dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        num_workers=4,
                                        shuffle=True,
                                        collate_fn=dataset.collate_fn)
    for data in dataloader:
        img,label,path = data
        # print(img,label,path)

        # break
    # img = cv2.imread('/home/autocore/work/yolov3_darknet/lishui/images/1599039225trans4.png')
    # newimg = letter_box(img,desired_size=416)
    # print(newimg.shape)