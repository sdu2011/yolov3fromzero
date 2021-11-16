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
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    yolov3net = Yolov3('cfg/yolov3.cfg')
    yolov3net.to(device).train()
    loss = YoloLoss(yolov3net)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, yolov3net.parameters()), lr=1e-4, weight_decay=5e-4)

    for epoch in range(1000):
        for data in dataloader:
            imgs,labels = data
            imgs = imgs.to(device)
            imgs = imgs.float()/255.
            # print(imgs.shape,labels.shape)
            # print(labels[:,0])
            yolo_out = yolov3net(imgs)
            # print([out.shape for out in yolo_out])

            lconf,lx,ly,lw,lh,lcls = loss.compute_loss(yolo_out,labels)
            total_loss = lconf + lx + ly + lw + lh + lcls
            print('total_loss={}'.format(total_loss))
            optimizer.zero_grad() #清空梯度
            total_loss.backward() #反向传播
            optimizer.step()      #更新参数
            