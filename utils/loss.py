from models.models import *

class YoloLoss(nn.Module):
    def __init__(self,model) :
        super(YoloLoss, self).__init__()
        self.model = model

    def compute_loss(self,yolo_outs,targets):
        """
        yolo_outs:list of Tensor [[batch,3,13,13,85],[batch,3,26,26,85],[batch,3,52,52,85]] 85:xywh+conf+class
        targets:[gt_box_num,6] image,cls,xywh
        """
        FT = torch.cuda.FloatTensor if yolo_outs[0].is_cuda else torch.FloatTensor
        lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

        BCEconf = nn.BCEWithLogitsLoss(reduction='mean')
        BCEcls = nn.BCEWithLogitsLoss(reduction='mean')
        #比sigmoid + BCELoss更稳定一些. 所以对conf和cls,就不对yolo_out求sigmoid后再求BCELoss了
        MSELoss = nn.MSELoss(reduction='mean')


        mask,gt_xywhc = self.match_gtbox_to_yololayer(targets,yolo_outs)
        #mask [gt_box_num,N] N:i(layer idx),a(anchor idx),grid_x,grid_y
        # print('mask={}'.format(mask))

        for i,yolo_out in enumerate(yolo_outs):
            mask_obj = torch.zeros_like(yolo_out[...,0],dtype=torch.bool)
            #标识yolo_out的哪一个grid的哪一个anchor应该负责预测出目标  [batch,3,13,13,85]
            gt_x = torch.zeros_like(yolo_out[...,0])
            gt_y = torch.zeros_like(yolo_out[...,0])
            gt_w = torch.zeros_like(yolo_out[...,0])
            gt_h = torch.zeros_like(yolo_out[...,0])
            gt_c = torch.zeros_like(yolo_out[...,5:])

            for j,e in enumerate(mask):
                batch_idx = targets[j,0].long()
                #targets:[gt_box_num,6] image,cls,xywh
                
                layer_idx = e[0].long()
                anchor_idx = e[1].long()
                grid_x = e[2].long()
                grid_y = e[3].long()

                if layer_idx == i:  
                    mask_obj[batch_idx,anchor_idx,grid_x,grid_y] = True
                    gt_x[batch_idx,anchor_idx,grid_x,grid_y] = gt_xywhc[j,0] 
                    gt_y[batch_idx,anchor_idx,grid_x,grid_y] = gt_xywhc[j,1] 
                    gt_w[batch_idx,anchor_idx,grid_x,grid_y] = gt_xywhc[j,2] 
                    gt_h[batch_idx,anchor_idx,grid_x,grid_y] = gt_xywhc[j,3] 

                    c = gt_xywhc[j,4].long()
                    gt_c[...,c] = True

            """
            conf loss
            """
            conf_pre = yolo_out[...,4]
            pos_conf_loss, neg_conf_loss= FT([0]), FT([0])
            if mask_obj.sum() > 0:
                pos_conf_loss = BCEconf(conf_pre[mask_obj],mask_obj[mask_obj].float())
                lconf += pos_conf_loss
                #该预测出目标的位置要预测出目标 conf趋向1
            if((~mask_obj).sum() > 0):
                neg_conf_loss = BCEconf(conf_pre[~mask_obj],mask_obj[~mask_obj].float())
                lconf += neg_conf_loss
                #不该预测出目标的位置要预测出无目标. conf趋向0.
            # print('pos_conf_loss:{},neg_conf_loss:{}'.format(pos_conf_loss,neg_conf_loss))

            """
            box loss
            """
            if mask_obj.sum() > 0:
                yolo_layer_index = self.model.yolo_layers_index[i]
                yolo_layer = self.model.module_list[yolo_layer_index]
                pre_box = torch.clone(yolo_out)
                yolo_layer.decode(pre_box)
                
                pre_x = pre_box[...,0]
                lx += MSELoss(pre_x[mask_obj], gt_x[mask_obj])
                
                pre_y = pre_box[...,1]
                ly += MSELoss(pre_y[mask_obj], gt_y[mask_obj])
                
                pre_w = pre_box[...,2]
                lw += MSELoss(pre_w[mask_obj], gt_w[mask_obj])
                
                pre_h = pre_box[...,3]
                lh += MSELoss(pre_h[mask_obj], gt_h[mask_obj])

            """
            cls loss
            """
            if mask_obj.sum() > 0:
                pre_cls = yolo_out[...,5:]
                lcls += BCEcls(pre_cls[mask_obj],gt_c[mask_obj])

        
        print('lconf={},lx={},ly={},lw={},lh={},lcls={}'.format(lconf,lx,ly,lw,lh,lcls))

        return lconf,lx,ly,lw,lh,lcls

    def match_gtbox_to_yololayer(self,targets,yolo_outs,threshold=0.5):
        """
        targets:[gt_box_num,6] image,cls,xywh
        return: 
        mask [gt_box_num,N] N:i(layer idx),a(anchor idx),grid_x,grid_y
        gt_box:相应特征图上的box尺寸和位置. [gt_box_num,4] x,y,w,h,c
        """
        gt_box_num = targets.shape[0]
        mask = torch.zeros(gt_box_num,4)
        gt_xywhc = torch.zeros(gt_box_num,5)

        all_ious = []
        grid_xys = []
        for i,pre in enumerate(yolo_outs):
            yolo_layer_index = self.model.yolo_layers_index[i]
            yolo_layer = self.model.module_list[yolo_layer_index]
            anchor_boxes = yolo_layer.anchor_scale

            bs,anchor_num,grid_x,grid_y,_ = pre.shape
            gt_boxes = targets.clone()
            # 这里要特别注意python中深浅拷贝的问题.如果用gt_boxes = targets,处理后实际上把targets和gt_boxes都更改掉了.
            gt_boxes[:,2] = gt_boxes[:,2]*grid_x
            gt_boxes[:,3] = gt_boxes[:,3]*grid_y
            gt_boxes[:,4] = gt_boxes[:,4]*grid_x
            gt_boxes[:,5] = gt_boxes[:,5]*grid_y
            # print('box size on layer{}:{}'.format(i,gt_boxes[0,2:]))
            #转换到当前特征图上的绝对位置和大小

            ious = self.wh_iou(gt_boxes[...,4:],anchor_boxes)
            #在当前特征图上计算gt box和anchor box的iou. [gt_box_num,3]

            all_ious.append(ious)

            grid_xy = gt_boxes[:,2:4].long()
            grid_xys.append(grid_xy)
        
        #处理所有特征图上的iou. 将gt box交给iou最高的anchor负责.
        all_ious = torch.cat(all_ious,dim=1)
        # print('all_ious={}'.format(all_ious))
        # [gt_box_num,9]
        value,indices = all_ious.max(dim=1)
        for i in range(gt_box_num):
            layer_idx = indices[i] // 3
            # /是精确除法  //是向下取整除法
            anchor_idx = indices[i] % 3
            mask[i,0] = layer_idx
            mask[i,1] = anchor_idx
            mask[i,2] = grid_xys[layer_idx][i,0]
            mask[i,3] = grid_xys[layer_idx][i,1]

            grid_x,grid_y = self.model.get_grid_num(layer_idx)
            gt_xywhc[i,0] = targets[i,2] * grid_x
            gt_xywhc[i,1] = targets[i,3] * grid_y
            gt_xywhc[i,2] = targets[i,4] * grid_x
            gt_xywhc[i,3] = targets[i,5] * grid_y
            gt_xywhc[i,4] = targets[i,1]

        return mask,gt_xywhc

    def wh_iou(self,wh1, wh2):
        # print('wh1={}'.format(wh1))
        # print('wh2={}'.format(wh2))

        # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
        # wh1 = wh1[:, None]  # [N,1,2]
        wh1 = wh1.unsqueeze(1) # [N,1,2]
        # print(wh1.shape)
        # wh2 = wh2[None]  # [1,M,2]
        wh2 = wh2.unsqueeze(0) # [1,M,2]
        # print('wh1 shape={}'.format(wh1.shape))
        # print('wh2 shape={}'.format(wh2.shape))
        # print('min={}'.format(torch.min(wh1,wh2).shape))
        inter = torch.min(wh1, wh2).prod(dim=2)  # [N,M]
        #torch.min用法见https://www.cnblogs.com/sdu20112013/p/11731741.html

        return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


if __name__ == '__main__':
    yolov3net = Yolov3('cfg/yolov3.cfg')
    l = YoloLoss(yolov3net)
    wh1 = torch.randn(3,2)
    wh2 = torch.randn(10,2)
    ious = l.wh_iou(wh1,wh2)
    print(ious.shape)



    




