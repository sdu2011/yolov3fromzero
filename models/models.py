import torch.nn as nn
from utils.parse_config import *
from layers import *

def create_modules(module_defs):
    input_channels = [3]
    
    module_list = nn.ModuleList() 
    yolo_index = -1
    yolo_stride = [32,16,8] 
    #3个yolo层的下采样倍数
    input_img_size = (416,416)

    record_layers = []
    # 记录下特定layer的输出.供route和shortcut使用.

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'net':
            width = mdef['width']
            height = mdef['height']
            input_img_size = (height,width)
            continue
        
        # filters = 0
        if mdef['type'] == 'convolutional':
            # print(mdef)
            bn = mdef['batch_normalize']
            k = mdef['size']
            filters = mdef['filters']
            stride = mdef['stride']

            modules.add_module('Conv2d', nn.Conv2d(in_channels=input_channels[-1],
                                        out_channels=filters,
                                        kernel_size=k,
                                        stride=stride,
                                        padding=k // 2 if mdef['pad'] else 0,
                                        bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(negative_slope=0.01,inplace=True))
        elif mdef['type'] == 'shortcut':
            from_layers = mdef['from']
            filters = input_channels[-1]
            record_layers.extend([i + l if l < 0 else l for l in from_layers])
            modules = FeatureShortcut(from_layers=from_layers)
        elif mdef['type'] == 'route':
            """
            [route]
            layers = -1, 61
            """
            layers = mdef['layers']
            filters = 0
            #更新route后的channel数目
            if len(layers) == 1:
                filters = input_channels[i + layers[0]]
            else:
                for layer in layers:
                    if layer < 0:
                        filters += input_channels[i + layer]
                    else:
                        filters += input_channels[layer + 1]
                        # +1是因为第一个input_channel=3.
            # print('filters={}'.format(filters))

            modules = FeatureConcat(layers)
            record_layers.extend([i + l if l < 0 else l for l in layers])
        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=mdef['stride'])
        elif mdef['type'] == 'maxpool':
            k = mdef['size']
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k,stride=stride,padding=(k-1)//2)         
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool
        elif mdef['type'] == 'yolo':
            print('yolo layer,i={}'.format(i))
            yolo_index += 1
            mask = mdef['mask']
            anchors = mdef['anchors']
            yolo_anchors = anchors[mask]
            print(yolo_anchors)
            cls_num = mdef['classes']
            modules = YoloLayer(
                        input_img_size,
                        anchors = yolo_anchors,
                        cls_num=cls_num,
                        stride = yolo_stride[yolo_index],
                        yolo_index=yolo_index)
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        module_list.append(modules)
        
        input_channels.append(filters)
        #依次存储每一层的输入的channel数量.

    # print('module_list=',module_list)
    return module_list

class Yolov3(nn.Module):
    def __init__(self,cfg):
        super(Yolov3, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list = create_modules(self.module_defs)

        # print(self.module_list)

    def forward(self, x):
        yolo_out = []
        out_every_layer = []
        # 记录每一层的输出.供route和shortcut使用.
        for i,module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name  == 'FeatureConcat':
                layers = module.layers
                # print('route from_layers={}'.format(layers))
                features_need_concat = [out_every_layer[layer] for layer in layers]
                x = module(features_need_concat)
                # print('FeatureConcat:x output shape={}'.format(x.shape))
            elif name == 'FeatureShortcut':
                from_layers = module.from_layers[0]
                x = module(x,out_every_layer[from_layers])
                # print('FeatureShortcut:x output shape={}'.format(x.shape))
            elif name == 'YoloLayer':
                yolo_out.append(module(x))
            else:
                # print('conv:x input shape={}'.format(x.shape))
                x = module(x)
                # print('conv:x output shape={}'.format(x.shape))

            out_every_layer.append(x)


        print('training:{}'.format(self.training))
        # if self.training:
        #     return yolo_out
        # else:


        return yolo_out
        
if __name__ == '__main__':
    yolov3net = Yolov3('cfg/yolov3.cfg')
    yolov3net.eval()
    # print(yolov3net.module_defs)
    input = torch.randn((1,3,416,416))
    yolo_out=yolov3net(input)
    print([out.shape for out in yolo_out])

    
