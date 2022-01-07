from models.models import *
from datasets.dataset import *
from utils.loss import *
from utils.utils import *
import time
import argparse
import test
import torch.nn.functional as F
import math

parser = argparse.ArgumentParser()
parser.add_argument('-resume', default=False, help='resume training flag')
parser.add_argument('-epochs', type=int, default=0, help='resume from number of epochs,only useful when resume is True')
parser.add_argument('-batchsize', type=int,default=96, help='training batch size')
parser.add_argument('-cfg', type=str,default='cfg/yolov3.cfg', help='training cfg')
parser.add_argument('-traintxt', type=str,default='coco/trainval2017.txt', help='training txt')
parser.add_argument('-testtxt', type=str,default='coco/val2017.txt', help='testing txt')
parser.add_argument('-model_input_size', type=int,default=416, help='model_input_size')
parser.add_argument('-conf_thre', type=float,default=0.7, help='confidence threshold')
parser.add_argument('-iou_thre', type=float,default=0.5, help='iou threshold')
parser.add_argument('-cls_thre', type=float,default=0.7, help='class threshold')
parser.add_argument('-cls_names_path', type=str,default='coco/names', help='class names file')
parser.add_argument('-conf_loss_weights', type=int,default=200, help='conf loss weights')
parser.add_argument('-negconf_loss_weights', type=int,default=5, help='neg conf loss weights')
parser.add_argument('-cls_loss_weights', type=int,default=50, help='cls loss weights')
parser.add_argument('-xy_loss_weights', type=int,default=5, help='xy loss weights')
parser.add_argument('-wh_loss_weights', type=int,default=1, help='wh loss weights')
parser.add_argument('-use_mosaic', action='store_true',help='use mosaic data augmentation')
parser.add_argument('-use_multi_scale', action='store_true',help='scale when training')
#cmd line加-use_mosaic则将use_mosaic设置为true,否则设置为false
parser.add_argument('-optimizer',type=str,default='Adam',help='optimizer type:Adam/Sgd')
parser.add_argument('-max_epochs',type=int,default='400',help='max epochs for training')
parser.add_argument('-gpus',type=str,default='0',help='gpu ids')

opt = parser.parse_args()
print(opt)
cls_names = load_classes(opt.cls_names_path)

if __name__ == '__main__':
    # Dataset
    # traintxt = '/home/autocore/work/yolov3_darknet/data/lishui/train.txt'
    root_dir=os.getcwd()
    traintxt = root_dir + '/' + opt.traintxt
    dataset = LoadImagesAndLabels(traintxt,cls_names,imgsize=opt.model_input_size,aug=True,mosaic=opt.use_mosaic)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=opt.batchsize,
                                            num_workers=os.cpu_count(),
                                            shuffle=True,
                                            collate_fn=dataset.collate_fn)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    yolov3net = Yolov3(opt.cfg)

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    #设置多gpu训练. 这一句必须放在import torch前才有效

    yolov3net = nn.DataParallel(yolov3net)

    yolov3net = yolov3net.to(device)
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, yolov3net.parameters()), lr=1e-4, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(yolov3net.parameters(), lr=.001, momentum=.9, weight_decay=0.0005 * 0, nesterov=True)
    start_epoch = 0
    resume = opt.resume
    if resume:
        checkpoint_name = 'checkpoints/epoch{}.pt'.format(opt.epochs)
        checkpoint = torch.load(checkpoint_name)
        yolov3net.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(yolov3net.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('resume done from {}*********************'.format(checkpoint_name))  

        del checkpoint

    yolov3net.train()
    loss = YoloLoss(yolov3net)

    nb = len(dataloader)
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    max_epochs = 300
    lf = lambda x: (((1 + math.cos(x * math.pi / max_epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1  # see link below
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # 设置学习率衰减. lf(epoch)随着训练的进行,lf(epoch)从1逐渐减小.　学习率=初始学习率*lf(epoch)　

    mAP,best_mAP = 0.,0.
    now=datetime.datetime.now()
    log_name = './log/train_log_{}.txt'.format(now)
    f_log = open(log_name,'a+') 
    f_log.writelines('begin training********************\n')
    for epoch in range(start_epoch,opt.max_epochs): 
        print('epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            print('当前学习率:{}'.format(param_group['lr']))
        
        t0 = time.time()
        total_loss_list = []
        pt_conf_list,nt_conf_list,lx_list,ly_list,lw_list,lh_list,lcls_list= [],[],[],[],[],[],[]

        mean_total_loss,mean_lconf_loss = 0.,0.
        for i,data in enumerate(dataloader):
            imgs,labels,_ = data
            imgs = imgs.to(device)
            imgs = imgs.float()/255.
            # print(imgs.shape,labels.shape)
            # print(labels[:,0])

            if opt.use_multi_scale:
                #改变模型输入大小
                origin_size = opt.model_input_size
                min_size = int(origin_size * 0.77)
                max_size = int(origin_size * 1.2)

                changed_size = random.randrange(min_size,max_size)
                r = int(changed_size/32)
                changed_size = r * 32
                imgs = F.interpolate(imgs, size=changed_size, mode='bilinear', align_corners=False)
                print('change model input from {} to {}'.format(origin_size,changed_size))

            yolo_out = yolov3net(imgs)
            # print([out.shape for out in yolo_out])

            lconf,lx,ly,lw,lh,lcls,pt_conf,nt_conf,statics = loss.compute_loss(yolo_out,labels,neg_weight=opt.negconf_loss_weights)
            total_loss = opt.conf_loss_weights * lconf + opt.xy_loss_weights * (lx + ly) + opt.wh_loss_weights * (lw + lh) + opt.cls_loss_weights * lcls
            # print('lconf={},lx={},ly={},lw={},lh={},lcls={}'.format(lconf.item(),lx.item(),ly.item(),lw.item(),lh.item(),lcls.item()))
            # print('img:{},total_loss={},lconf:{},pt_conf:{},nt_conf:{},lx={},ly={},lcls={}'.format(
            #     (1+i)*opt.batchsize,total_loss.item(),lconf.item(),pt_conf.item(),nt_conf.item(),lx.item(),ly.item(),lcls.item()))
            
            total_box_num,total_nobox_num,correct_posconf,correct_negconf,correct_x,correct_y = statics
            print('---total_box_num:{},posconf:{:.2f},,negconf:{:.2f},correct_x:{:.2f},correct_y:{:.2f}---'.\
                    format(total_box_num,\
                           correct_posconf/total_box_num,\
                           correct_negconf/total_nobox_num,\
                           correct_x/total_box_num,\
                           correct_y/total_box_num,\
                           )
                    )
            
            optimizer.zero_grad() #清空梯度
            total_loss.backward() #反向传播
            optimizer.step()      #更新参数

            total_loss_list.append(total_loss.item())
            pt_conf_list.append(pt_conf.item())
            nt_conf_list.append(nt_conf.item())
            lx_list.append(lx.item())
            ly_list.append(ly.item())
            lcls_list.append(lcls.item())
            lw_list.append(lw.item())
            lh_list.append(lh.item())

            mean_total_loss = np.mean(np.array(total_loss_list))
            mean_pt_conf_loss = np.mean(np.array(pt_conf_list))
            mean_nt_conf_loss = np.mean(np.array(nt_conf_list))
            mean_lx_loss = np.mean(np.array(lx_list))
            mean_ly_loss = np.mean(np.array(ly_list))
            mean_lcls_loss = np.mean(np.array(lcls_list))
            mean_lw_loss = np.mean(np.array(lw_list))
            mean_lh_loss = np.mean(np.array(lh_list))
            print('*****img:{},total loss:{:.3f},pt_conf:{:.3f},nt_conf:{:.3f},lx:{:.3f},ly:{:.3f},lcs:{:.3f},lw:{:.3f},lh:{:.3f}'.format((1+i)*opt.batchsize,\
                                                    mean_total_loss,\
                                                    mean_pt_conf_loss,\
                                                    mean_nt_conf_loss,\
                                                    mean_lx_loss,\
                                                    mean_ly_loss,\
                                                    mean_lcls_loss,\
                                                    mean_lw_loss,\
                                                    mean_lh_loss        
                                                    )
            )

        
        # Update scheduler
        scheduler.step()
        
        t1 = time.time()        
        print('epoch{} train for {}'.format(epoch,(t1-t0)))

        checkpoint = {'epoch':epoch,
                      'model':yolov3net.state_dict(),
                      'optimizer':optimizer.state_dict()}
        if epoch % 1 == 0:
            checkpoint_name = 'checkpoints/epoch{}.pt'.format(epoch)
            torch.save(checkpoint,checkpoint_name)
        
        #写日志
        now=datetime.datetime.now()
        # f_log.writelines('{},epoch:{},total_loss:{},pt_conf:{},nt_conf:{},lx={},ly={},lw={},lh={},lcls={}\n'. \
        #         format(str(now),epoch,total_loss.item(),pt_conf.item(),nt_conf.item(),lx.item(),ly.item(),lw.item(),lh.item(),lcls.item()))
        f_log.writelines('{},epoch:{},total_loss:{},pt_conf:{},nt_conf:{},lx:{},ly:{},lcls:{},lw:{},lh:{}\n'.\
            format(str(now),epoch,mean_total_loss,mean_pt_conf_loss,
            mean_nt_conf_loss,mean_lx_loss,mean_ly_loss,mean_lcls_loss,mean_lw_loss,mean_lh_loss))
        f_log.flush()

        #测试
        if epoch % 1 == 0:
            mAP = test.test(opt.cfg,opt.testtxt,checkpoint_name,opt.model_input_size,
                      opt.conf_thre,opt.iou_thre,opt.cls_thre,cls_names)

            if mAP > best_mAP:
                print('************************mAP:{},best_mAP:{}'.format(mAP,best_mAP))
                best_mAP = mAP
                checkpoint_name = 'best.pt'
                torch.save(checkpoint,checkpoint_name)

            f_log.writelines('epoch:{},mAP={},best_mAP={}\n'.format(epoch,mAP,best_mAP))
            f_log.flush()

        # if lconf.item() < 0.01 or total_loss.item() < 0.1 or mAP > 0.4:
        if mAP > 0.6:
            break
    f_log.close()

    torch.cuda.empty_cache()