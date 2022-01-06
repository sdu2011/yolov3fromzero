from models.models import *
from datasets.dataset import *
from utils.utils import *


def test(cfg,testtxt,checkpoint_name,model_input_size,conf_thre,iou_thre,cls_thre,cls_names,bs=32):
    # testtxt = '/home/autocore/work/yolov3fromzero/cfg/test.txt'
    start=datetime.datetime.now()
    print('test begin,testtxt:{}'.format(testtxt))
    
    dataset = LoadImagesAndLabels(testtxt,cls_names,imgsize=model_input_size,aug=False,debug=False)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=bs,
                                            num_workers=8,
                                            shuffle=False,
                                            collate_fn=dataset.collate_fn)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    yolov3net = Yolov3(cfg)

    if checkpoint_name.endswith('.weights'):
        load_darknet_weights(yolov3net, checkpoint_name)
        # pass
    elif checkpoint_name.endswith('.pt'):
        checkpoint = torch.load(checkpoint_name)
        yolov3net.load_state_dict(checkpoint['model'])
    print('****************model weights load done********************')
    #加载模型

    yolov3net.to(device).eval()
    APs,Recalls,Precisions=[],[],[]
    for i,data in enumerate(dataloader):
        imgs,labels,imgs_path = data
        imgs = imgs.to(device)
        imgs = imgs.float()/255.

        print('image{}'.format(i * bs))
        # print('imgs_path:{},labels:{}'.format(imgs_path,labels))

        img_size = imgs.shape[2]

        with torch.no_grad():
            yolo_outs = yolov3net(imgs)
            print('yolo_out shape={}'.format(yolo_outs[0].shape))
            
            #output[j,j,:]:1.53957062138943e-05

            detections = post_process(imgs,imgs_path,yolo_outs,img_size,conf_thre,iou_thre,cls_thre,cls_names)
            metric(APs,Recalls,Precisions,imgs_path,detections,labels,img_size,iou_thre,cls_thre)
    print('mAP={}'.format(np.mean(APs)))
    print('recall={}'.format(np.mean(Recalls)))
    print('precision={}'.format(np.mean(Precisions)))
    end=datetime.datetime.now()
    print('耗时{}'.format(end - start))

    return np.mean(APs)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default='checkpoints/epoch200.pt', help='model name')
    parser.add_argument('-model_input_size', type=int, default=416, help='model input size')
    parser.add_argument('-testtxt', type=str,default='coco/debug_test2017.txt', help='testing txt')
    parser.add_argument('-cfg', type=str,default='cfg/yolov3.cfg', help='training cfg')
    parser.add_argument('-conf_thre', type=float,default=0.7, help='confidence threshold')
    parser.add_argument('-iou_thre', type=float,default=0.5, help='iou threshold')
    parser.add_argument('-cls_thre', type=float,default=0.7, help='class threshold')
    parser.add_argument('-cls_names_path', type=str,default='coco/names', help='class names file')
    opt = parser.parse_args()
    print(opt.model_path)

    cls_names = load_classes(opt.cls_names_path)
    
    test(opt.cfg,opt.testtxt,opt.model_path,opt.model_input_size,opt.conf_thre,opt.iou_thre,opt.cls_thre,cls_names,bs=64)

    torch.cuda.empty_cache()



