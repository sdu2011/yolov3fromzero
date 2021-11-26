from models.models import *
from datasets.dataset import *
from utils.loss import *
import time
import argparse
import test

parser = argparse.ArgumentParser()
parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
parser.add_argument('-resume', default=False, help='resume training flag')
parser.add_argument('-batchsize', type=int,default=16, help='training batch size')

opt = parser.parse_args()
print(opt)

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
    yolov3net = Yolov3('cfg/yolov3_tlr.cfg')
    yolov3net = yolov3net.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, yolov3net.parameters()), lr=1e-4, weight_decay=5e-4)
    start_epoch = 0
    resume = opt.resume
    if resume:
        checkpoint_name = 'checkpoints/epoch{}.pt'.format(opt.epochs)
        checkpoint = torch.load(checkpoint_name)
        yolov3net.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(yolov3net.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('resume done*********************')  
    
    yolov3net.train()
    loss = YoloLoss(yolov3net)

    for epoch in range(start_epoch,400):    
        print('epoch {}'.format(epoch))
        t0 = time.time()
        for data in dataloader:
            imgs,labels,_ = data
            imgs = imgs.to(device)
            imgs = imgs.float()/255.
            # print(imgs.shape,labels.shape)
            # print(labels[:,0])
            yolo_out = yolov3net(imgs)
            # print([out.shape for out in yolo_out])

            lconf,lx,ly,lw,lh,lcls = loss.compute_loss(yolo_out,labels,neg_weight=5)
            total_loss = 2* lconf + 2 * lx + 2 * ly + lw + lh + lcls
            # print('lconf={},lx={},ly={},lw={},lh={},lcls={}'.format(lconf.item(),lx.item(),ly.item(),lw.item(),lh.item(),lcls.item()))
            print('total_loss={}'.format(total_loss.item()))
            optimizer.zero_grad() #清空梯度
            total_loss.backward() #反向传播
            optimizer.step()      #更新参数
        
        t1 = time.time()
        print('epoch{} train for {}'.format(epoch,(t1-t0)))

        checkpoint = {'epoch':epoch,
                      'model':yolov3net.state_dict(),
                      'optimizer':optimizer.state_dict()}
        if epoch % 5 == 0:
            checkpoint_name = 'checkpoints/epoch{}.pt'.format(epoch)
            torch.save(checkpoint,checkpoint_name)

            test.test(checkpoint_name)
        