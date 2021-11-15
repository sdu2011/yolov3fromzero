from models.models import *
from datasets.dataset import *
from utils.loss import *

if __name__ == '__main__':
    # Dataset
    traintxt = '/home/autocore/work/yolov3_darknet/data/lishui/train.txt'
    dataset = LoadImagesAndLabels(traintxt,imgsize=416)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=8,
                                            num_workers=4,
                                            shuffle=True,
                                            collate_fn=dataset.collate_fn)
    yolov3net = Yolov3('cfg/yolov3.cfg')
    yolov3net.train()
    loss = YoloLoss(yolov3net)
    for epoch in range(100):
        for data in dataloader:
            imgs,labels = data
            imgs = imgs.float()/255.
            # print(imgs.shape,labels.shape)
            # print(labels[:,0])
            yolo_out = yolov3net(imgs)
            # print([out.shape for out in yolo_out])

            lconf,lx,ly,lw,lh,lcls = loss.compute_loss(yolo_out,labels)
            total_loss = lconf + lx + ly + lw + lh + lcls
            print('total_loss={}'.format(total_loss))
            total_loss.backward()
            