from models.models import *
from datasets.dataset import *
from utils.utils import *

def test(checkpoint_name,conf_thre=0.9,iou_thre=0.6):
    testtxt = '/home/autocore/work/yolov3fromzero/cfg/test.txt'
    print('test begin,testtxt:{}'.format(testtxt))
    dataset = LoadImagesAndLabels(testtxt,imgsize=416)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1,
                                            num_workers=1,
                                            shuffle=True,
                                            collate_fn=dataset.collate_fn)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    yolov3net = Yolov3('cfg/yolov3_tlr.cfg')
    yolov3net = yolov3net.to(device)
    #
    checkpoint = torch.load(checkpoint_name)
    yolov3net.load_state_dict(checkpoint['model'])
    #加载模型

    # yolov3net.eval()
    yolov3net.eval()
    for data in dataloader:
        imgs,labels,imgs_path = data
        imgs = imgs.to(device)
        imgs = imgs.float()/255.

        # print('imgs_path:{},labels:{}'.format(imgs_path,labels))

        img_size = imgs.shape[2]

        with torch.no_grad():
            yolo_outs = yolov3net(imgs)
            # print('yolo_out shape={}'.format(yolo_outs[0].shape))
            
            detections = post_process(imgs,imgs_path,yolo_outs,img_size=img_size,conf_thre=conf_thre,iou_thre=iou_thre)
            metric(detections,labels,img_size)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default='checkpoints/epoch200.pt', help='model name')
    opt = parser.parse_args()
    print(opt.model_path)
    test(opt.model_path)





