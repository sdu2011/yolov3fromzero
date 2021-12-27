from torch._C import device
from models.models import *
from datasets.dataset import *
from utils.utils import *


def letter_box(img,desired_size,color=[114,114,114]):
    """
    把img resize到特定尺寸. 保持宽高比.
    desired_size : (h,w)

    return:尺寸为desired_size的img
    """
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

    delta_w = max(0,desired_size[1] - new_size[1])
    delta_h = max(0,desired_size[0] - new_size[0])
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    # print('top:{},bottom:{},left:{},right:{}'.format(top,bottom,left,right))
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_img

def resize_square(img, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2


def detect(dir_name,cfg,checkpoint_name,model_input_size,conf_thre,iou_thre,cls_thre,cls_names):
    start=datetime.datetime.now()

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    device = 'cpu'
    yolov3net = Yolov3(cfg)

    if checkpoint_name.endswith('.weights'):
        # load_darknet_weights(yolov3net, checkpoint_name)
        load_weights(yolov3net, checkpoint_name)
    elif checkpoint_name.endswith('.pt'):
        checkpoint = torch.load(checkpoint_name)
        yolov3net.load_state_dict(checkpoint['model'])
        del checkpoint
    print('****************model weights load done********************')
    #加载模型

    # yolov3net.eval()
    yolov3net = yolov3net.to(device)
    yolov3net.eval()

    imgs = os.listdir(dir_name)
    for img_path in imgs:
        print(img_path)
        full_path = '{}/{}'.format(dir_name,img_path)
        img = cv2.imread(full_path)
        # new_img = letter_box(img,(model_input_size,model_input_size))
        # cv2.imwrite('./test_imgs/test.jpg',new_img)
        # img, _, _, _ = resize_square(img, height=416, color=(127.5, 127.5, 127.5))
        new_img = img
        new_img = new_img[:,:,::-1] #bgr->rgb
        new_img = new_img.transpose( 2, 0, 1)  #hwc-chw
        new_img = np.ascontiguousarray(new_img)
        new_img = torch.from_numpy(new_img).to(device)
        new_img = new_img.float()/255.
        new_img = new_img.view(-1,3,model_input_size,model_input_size)

        imgs_path = [full_path]

        with torch.no_grad():
            print('fuck:{}'.format(new_img[0,2,325,18]))
            yolo_outs = yolov3net(new_img)
            detections = post_process(new_img,imgs_path,yolo_outs,model_input_size,conf_thre,iou_thre,cls_thre,cls_names)

    end=datetime.datetime.now()
    print('耗时{}'.format(end - start))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default='checkpoints/epoch200.pt', help='model name')
    parser.add_argument('-model_input_size', type=int, default=416, help='model input size')
    parser.add_argument('-dir_name', type=str,default='test_imgs', help='img_dir')
    parser.add_argument('-cfg', type=str,default='cfg/yolov3.cfg', help='training cfg')
    parser.add_argument('-conf_thre', type=float,default=0.7, help='confidence threshold')
    parser.add_argument('-iou_thre', type=float,default=0.5, help='iou threshold')
    parser.add_argument('-cls_thre', type=float,default=0.7, help='class threshold')
    parser.add_argument('-cls_names_path', type=str,default='coco/names', help='class names file')
    opt = parser.parse_args()
    print(opt)

    cls_names = load_classes(opt.cls_names_path)
    detect(opt.dir_name,opt.cfg,opt.model_path,opt.model_input_size,opt.conf_thre,opt.iou_thre,opt.cls_thre,cls_names)





