from models.models import *
from datasets.dataset import *
from utils.utils import *

if __name__ == '__main__':
    # Dataset
    testtxt = '/home/autocore/work/yolov3_darknet/data/lishui/test.txt'
    dataset = LoadImagesAndLabels(testtxt,imgsize=416)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=8,
                                            num_workers=4,
                                            shuffle=True,
                                            collate_fn=dataset.collate_fn)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    yolov3net = Yolov3('cfg/yolov3.cfg')
    yolov3net = yolov3net.to(device)
    #
    checkpoint = torch.load('checkpoints/epoch90.pt')
    yolov3net.load_state_dict(checkpoint['model'])
    #加载模型

    # yolov3net.eval()
    yolov3net.eval()
    for data in dataloader:
        imgs,labels,imgs_path = data
        imgs = imgs.to(device)
        imgs = imgs.float()/255.

        with torch.no_grad():
            yolo_outs = yolov3net(imgs)
            print('yolo_out shape={}'.format(yolo_outs[0].shape))
            
            post_process(imgs,imgs_path,yolo_outs)




