from models.models import *
from datasets.dataset import *
from utils.loss import *

if __name__ == '__main__':
    # Dataset
    traintxt = '/home/autocore/work/yolov3_darknet/data/lishui/train.txt'
    dataset = LoadImagesAndLabels(traintxt,imgsize=416)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=16,
                                            num_workers=4,
                                            shuffle=True,
                                            collate_fn=dataset.collate_fn)
    yolov3net = Yolov3('cfg/yolov3.cfg')
    for epoch in range(1):
        yolov3net.train()

        for data in dataloader:
            imgs,labels = data
            imgs = imgs.float()/255.
            print(imgs.shape,labels.shape)
            # print(labels[:,0])
            yolo_out = yolov3net(imgs)
            print([out.shape for out in yolo_out])

            compute_loss(yolo_out,labels)
            
            break