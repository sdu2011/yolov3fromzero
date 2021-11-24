import numpy as np
import cv2
import torch
import os

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
        print('ovr:{}'.format(ovr))
        #计算所有Box与当前score最高的box的iou

        inds = np.where(ovr <= thresh)[0]
        #过滤掉与当前box的iou过高的box. 保留剩下的box
        order = order[inds + 1]
        #更新待处理box

    return keep

def plot_one_box(x, img, color=(255,0,0), label=None, line_thickness=None):
    # print('img shape:{},x:{}'.format(img.shape,x))
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # print('c1:{},c2:{}'.format(c1,c2))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.rectangle(img, c1, c2, color)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def post_process(imgs,imgs_path,yolo_outs,img_size=416,conf_thre=0.9,iou_thre=0.3):
    output = torch.cat(yolo_outs,dim=1)
    bs = output.shape[0]
    for i in range(bs):
        img_name = (imgs_path[i]).split('/')[-1]
        # print(img_name)
        cur_path =  os.path.abspath(os.path.dirname(__file__))
        full_name = '{}/../out_imgs/{}'.format(cur_path,img_name)
        # print(full_name) 

        pre_conf = output[i,...,4]
        mask =  pre_conf > conf_thre
        # print('msak={}'.format(mask.shape))
        boxes = output[i][mask]
        # print('boxes:{}'.format(boxes))

        box_num = boxes.shape[0]
        if box_num > 0:
            print('box_num={}'.format(box_num))
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
            print('boxes num={}'.format(boxes.shape[0]))
            keep = nms(boxes, iou_thre)
            
            final_boxes = boxes[keep]
            print('final_boxes num={}'.format(final_boxes.shape[0]))

            cv_img = imgs[i,...].cpu().numpy()
            cv_img = cv_img.transpose(1,2,0)[...,::-1]
            cv_img = np.ascontiguousarray(cv_img)
            cv_img = (cv_img * 255).astype('uint8')
            ##chw rgb --> hwc bgr
            
            final_box_num = final_boxes.shape[0]
            for i in range(final_box_num):
                box = final_boxes[i,0:4]
                plot_one_box(box, cv_img, color=(255,0,0), label=None, line_thickness=None)

            cv2.imwrite(full_name,cv_img)


if __name__ == '__main__':
    pass