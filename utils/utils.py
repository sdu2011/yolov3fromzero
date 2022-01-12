import numpy as np
import cv2
import torch
import os
from pathlib import Path

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # print('latest recall:{},latest precision:{}'.format(recall[-1],precision[-1]))
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def nms(dets, thresh):
    """
    dets:[n,5] 5:lt_x,lt_y,rb_x,rb_y,conf 
    以四个box为例.
    1. 按score排序:box0,box3,box2,box1
    2. box0 score最高,依次计算box3,box2,box1与box0的iou,iou=[0,1,100](单位%)
    3. 根据thresh做过滤. box3,box2被保留.即box3,box2不与box0重叠.
    4. 对box3,box2开始处理.重复上述过程。
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # print('scores shape:{}'.format(scores.shape))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    #对每一个box按score排序.得到box的id.

    keep = []
    while order.size > 0:
        # print('order size={}'.format(order.size))
        i = order[0]
        # 当前处理的box中score最高的box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # print('ovr shape:{}'.format(ovr.shape))
        # print('ovr:{}'.format(ovr))
        #计算所有Box与当前score最高的box的iou

        inds = np.where(ovr <= thresh)[0]
        #过滤掉与当前box的iou过高的box. 保留剩下的box
        order = order[inds + 1]
        #更新待处理box

    return keep

def plot_one_box(x, img, color=(255,0,0), labels=None, line_thickness=None):
    # print('img shape:{},x:{}'.format(img.shape,x))
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # print('c1:{},c2:{}'.format(c1,c2))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.rectangle(img, c1, c2, color)
    if len(labels) > 0:
        for label in labels:
            label = str(label)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 8, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 8, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_one_box_on_origin_img(x, origin_cv_img, model_input_size=416,color=(255,0,0), labels=None, line_thickness=None):
    """
    x代表的Box的位置是相对于做了处理的输入模型的图片的.
    origin_cv_img是未做处理的原始图片.
    """
    h,w = origin_cv_img.shape[0],origin_cv_img.shape[1]
    if x[0] > w or x[1] > h or x[2] > w or x[3] > h:
        return
    #由于模型的输出并未对w,h做限制.所以要防止w,h预测错误的情况.

    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    center_x,center_y = (c1[0] + c2[0])/2,(c1[1] + c2[1])/2
    box_w,box_h = c2[0] - c1[0],c2[1]-c1[1]
    # print('center_x:{},center_y:{},box_w:{},box_h:{}'.format(center_x,center_y,box_w,box_h))

    h,w,c = origin_cv_img.shape
    if h/model_input_size > w/model_input_size:
        #在w方向上做padding
        r = model_input_size / h
        new_w = int(w * r)
        padding_x = (w - new_w)/2

        center_x = center_x - padding_x

        center_x_in_origin_img = center_x * w/new_w 
        center_y_in_origin_img = center_y * w/new_w
        box_w_in_origin_img = box_w/r
        box_h_in_origin_img = box_h/r
    else:
        #在h方向上做padding
        r = model_input_size / w
        
        new_h = int(h * r)
        padding_y = (model_input_size - new_h)/2
        
        center_y = center_y - padding_y

        center_x_in_origin_img = center_x * h/new_h 
        center_y_in_origin_img = center_y * h/new_h
        box_w_in_origin_img = box_w/r
        box_h_in_origin_img = box_h/r

    c1 = (int(center_x_in_origin_img-box_w_in_origin_img/2),int(center_y_in_origin_img-box_h_in_origin_img/2))
    c2 = (int(center_x_in_origin_img+box_w_in_origin_img/2),int(center_y_in_origin_img+box_h_in_origin_img/2))

    tl = line_thickness or round(0.002 * (origin_cv_img.shape[0] + origin_cv_img.shape[1]) / 2) + 1  # line/font thickness
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # print('c1:{},c2:{}'.format(c1,c2))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.rectangle(origin_cv_img, c1, c2, color)
    if len(labels) > 0:
        for label in labels:
            label = str(label)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 8, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(origin_cv_img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(origin_cv_img, label, (c1[0], c1[1] - 2), 0, tl / 8, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def post_process(imgs,imgs_path,yolo_outs,img_size,conf_thre,iou_thre,cls_prob,cls_names):
    """
    conf_thre: 超过该值则认为有目标
    iou_thre： 超过该值则认为两个box需要做nms
    cls_prob: 类别概率超过该值才认为是该类别
    return [[boxes of img1],[boxes of img2],....] 
    """
    output = torch.cat(yolo_outs,dim=1)

    print(output[0,1,:4])

    bs = output.shape[0]

    # detections = [[]] * bs
    #!!!!超级大坑  detections[x]的每一个元素的地址都是一样的 可以用id(detections[x])打印查看
    
    detections=[]

    for i in range(bs):
        origin_cv_img = cv2.imread(imgs_path[i])
        img_name = (imgs_path[i]).split('/')[-1]
        # print('post_process:{}'.format(imgs_path[i]))
        cur_path =  os.path.abspath(os.path.dirname(__file__))
        full_name = '{}/../out_imgs/{}'.format(cur_path,img_name)
        # print(full_name) 

        pre_conf = output[i,...,4]
        
        mask =  pre_conf > conf_thre
        # print('msak={}'.format(mask.shape))
        boxes = output[i][mask]
        # print('boxes:{}'.format(boxes))

        box_num = boxes.shape[0]
        # print('box_num={}'.format(box_num))
        if box_num > 0:
            # print('box_num={}'.format(box_num))
            box_center_x = boxes[...,0]
            box_center_y = boxes[...,1]

            box_lt_x = box_center_x - boxes[...,2]/2
            box_lt_y = box_center_y - boxes[...,3]/2
            box_rb_x = box_center_x + boxes[...,2]/2
            box_rb_y = box_center_y + boxes[...,3]/2

            boxes[...,0] = box_lt_x
            boxes[...,1] = box_lt_y
            boxes[...,2] = box_rb_x
            boxes[...,3] = box_rb_y

            boxes = boxes.cpu().numpy()
            # print('predict boxes num={}'.format(boxes.shape[0]))
            keep = nms(boxes, iou_thre)
            
            final_boxes = boxes[keep]
            # print('after nms final_boxes num={}'.format(final_boxes.shape[0]))
            detections.append([final_boxes])

            # print(imgs.shape)
            cv_img = imgs[i,...].cpu().numpy()
            cv_img = cv_img.transpose(1,2,0)[...,::-1]
            cv_img = np.ascontiguousarray(cv_img)
            cv_img = (cv_img * 255).astype('uint8')
            #chw rgb --> hwc bgr
            
            final_box_num = final_boxes.shape[0]
            for i in range(final_box_num):
                box = final_boxes[i,0:4]
                # print('plot box on img************')

                pre_cls_prob = final_boxes[i,5:]
                det_cls_idx = np.where(pre_cls_prob > cls_prob)[0].tolist()
                
                if len(det_cls_idx) == 0:
                    pre_cls_prob = pre_cls_prob.tolist()
                    det_cls_idx = [pre_cls_prob.index(max(pre_cls_prob))]
                #如果没有满足的cls_prob,则取prob最大的作为类别.


                det_cls_name = [cls_names[idx] for idx in det_cls_idx]
                # print('det_cls_idx:{},name:{}'.format(det_cls_idx,det_cls_name))

                # plot_one_box(box, cv_img, color=(255,0,0), labels=det_cls_idx, line_thickness=None)
                plot_one_box_on_origin_img(box, origin_cv_img,model_input_size=img_size, color=(255,0,0), labels=det_cls_name, line_thickness=None)

            # cv2.imwrite(full_name,cv_img)
            cv2.imwrite(full_name,origin_cv_img)
            print('save result to:{}'.format(full_name))
            
    
    return detections

def bbox_iou(box1, box2, x1y1x2y2=True):
    # if len(box1.shape) == 1:
    #    box1 = box1.reshape(1, 4)

    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def metric(APs,Recalls,Precisions,imgs_path,detections,labels,img_size,iou_thre,cls_thre):
    """
    统计一个batch内的AP,添加到APs.
    labels: [n,6] 6:img_idx,xywhc
    iou_thre: 与真实框iou超过该值则认为匹配成功
    cls_thre: 超过该值则认为类别判断正确
    """
    bs = len(detections)
    # print('bs={}'.format(bs))
    for i in range(bs):  
        print('metric:img {}'.format(imgs_path[i]))
        correct = []

        mask = (labels[:,0] == i) 
        label = labels[mask]  
        # print('label :{}'.format(label))
        #属于这张图的label

        gt_boxes = label[:,2:] * img_size
        # print('尺寸:gt_boxes :{}'.format(gt_boxes))
        gt_boxes_clone = torch.zeros_like(gt_boxes)
        gt_boxes_clone[:,0] = gt_boxes[:,0] - gt_boxes[:,2]/2
        gt_boxes_clone[:,1] = gt_boxes[:,1] - gt_boxes[:,3]/2
        gt_boxes_clone[:,2] = gt_boxes[:,0] + gt_boxes[:,2]/2
        gt_boxes_clone[:,3] = gt_boxes[:,1] + gt_boxes[:,3]/2
        #转换成角点的表达形式

        # print('角点:gt_boxes:{}'.format(gt_boxes_clone))

        detection = detections[i]
        if len(detection) == 0:
            print('no detection result')
            APs.append(0)
        else:
            detected_boxes = detection[0]
            # xywh+conf+c
            idx = np.argsort(detected_boxes[...,4])[::-1]
            detected_boxes = detected_boxes[idx]
            #对检测框按照conf降序排列
            """
            挨个处理每一个预测框b,计算gt boxes和b的iou,取iou最大的gtbox_i,认为b是gtbox_i对应的预测框.
            """

            matched_gt_box_idx=[]
            #已经匹配了的gt box就不可以继续匹配了
            for det_box in detected_boxes:
                det_box = torch.from_numpy(det_box).float()
                det_box = det_box.view(1,-1)
                #计算gt box和det_box的iou
                # print('det_box:{}'.format(det_box))
                # print('gt_boxes:{}'.format(gt_boxes_clone))
                ious=bbox_iou(det_box,gt_boxes_clone)
                # print('ious :{}'.format(ious))

                best_gt_i = np.argmax(ious)
                

                iou_matched = ious[best_gt_i]
                # print('匹配iou为:{}'.format(iou_matched))
                iou_satisfied = iou_matched > iou_thre
                
                # print('真值:gt_boxes :{}'.format(gt_boxes))
                gt_box_cls = int(label[best_gt_i,1])
                pre_cls_prob = det_box[0,5:]
                #注意det_box[0,5:]与det_box[:,5:]的不同　会造成shape不同
                # print('预测概率为:{}，真实类别为:{}'.format(det_cls,gt_box_cls))
                det_cls_idx = np.where(pre_cls_prob > cls_thre)[0].tolist()
                # print('cls_thre:{},预测类别为:{}'.format(cls_thre,det_cls_idx))
                if len(det_cls_idx) == 0:
                    pre_cls_prob = pre_cls_prob.numpy().tolist()
                    det_cls_idx = [pre_cls_prob.index(max(pre_cls_prob))]
                    # print('cls_thre:{},预测类别为:{}'.format(cls_thre,det_cls_idx))

                cls_correct = gt_box_cls in det_cls_idx
                # print('类别预测正确:{},真值:{},预测值:{}'.format(cls_correct,gt_box_cls,det_cls_idx))
                # print('iou匹配成功:{}'.format(iou_satisfied))
                

                if iou_satisfied and cls_correct and (best_gt_i not in matched_gt_box_idx):
                # if cls_correct:
                    correct.append(1)
                    matched_gt_box_idx.append(best_gt_i)
                else:
                    # print('iou_matched:{}'.format(iou_matched))
                    # print('det_cls_idx:{}'.format(det_cls_idx))
                    correct.append(0)

            # print('correct:{}'.format(correct))

            true_positives = np.array(correct) 
            true_positives = np.cumsum(true_positives)
            recall = true_positives / gt_boxes.shape[0]
            #更新recall列表

            pre_counts = np.arange(1,1+len(true_positives))
            #生成猜测次数列表 从1到n
            precision = true_positives/pre_counts
            #更新precision列表

            AP = compute_ap(recall,precision)
            APs.append(AP)
            Recalls.append(recall[-1])
            Precisions.append(precision[-1])

            print('latest recall:{},latest precision:{}'.format(recall[-1],precision[-1]))
            if recall[-1] < 0.6 or precision[-1]< 0.6:
                # print('latest recall:{},latest precision:{}'.format(recall[-1],precision[-1]))
                print('low recall/precision img {}'.format(imgs_path[i]))
            # print('img{} in this batch,recall:{},precision:{},AP:{}'.format(i,recall[-1],precision[-1],AP))
    
    # print('************mAP={}*****************'.format(np.mean(APs)))

    return 

def load_darknet_weights(yolov3net, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(yolov3net.module_defs[1:cutoff], yolov3net.module_list[:cutoff])):
        # print(mdef)
        if mdef['type'] == 'convolutional':
            # print(module)
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw

def load_weights(yolov3_net, weights_path):
    """Parses and loads the weights stored in 'weights_path'"""

    # Open the weights file
    fp = open(weights_path, "rb")
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    header_info = header

    seen = header[3]
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(yolov3_net.module_defs[1:], yolov3_net.module_list)):
        print(module_def)
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize'] == 1:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

if __name__ == '__main__':
    pass