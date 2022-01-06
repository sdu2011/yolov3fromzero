import torch.nn as nn
from utils.parse_config import *
from layers import *

darknet = False
if darknet:
    def create_modules(module_defs):
        print('create model as origin darknet param!')
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
            
            if mdef['type'] == 'convolutional':
                bn = int(mdef['batch_normalize'])
                filters = int(mdef['filters'])
                kernel_size = int(mdef['size'])
                pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
                modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=input_channels[-1],
                                                            out_channels=filters,
                                                            kernel_size=kernel_size,
                                                            stride=int(mdef['stride']),
                                                            padding=pad,
                                                            bias=not bn))
                if bn:
                    modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
                if mdef['activation'] == 'leaky':
                    modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))
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
else:    
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
                                            padding= k // 2 if mdef['pad'] else 0,
                                            bias=not bn))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
                    # modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters)) #原始的darknet weights的bn是不带动量的
                if mdef['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(negative_slope=0.01,inplace=True))
                    # modules.add_module('activation', nn.LeakyReLU(0.1)) #原始的darkenet weights用的是0.1
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
        # for m in self.module_defs:
        #     print(m)
        self.module_list = create_modules(self.module_defs)
        self.yolo_layers_index = [i for i, m in enumerate(self.module_list) \
            if m.__class__.__name__ == 'YoloLayer']
        self.yolo_layers_num = len(self.yolo_layers_index)
        #yolo layer的层数

        # print(self.module_list)

    def get_grid_num(self,i):
        """
        i:第几个yolo层
        """
        yolo_layer_index = self.yolo_layers_index[i]
        yolo_layer = self.module_list[yolo_layer_index]

        return yolo_layer.get_grid()

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
                b,c,h,w=0,1,2,3
                # print('yolo input:x[{},{},{},{}]:{}'.format(b,c,h,w,x[b,c,w,h]))
                out = module(x)
                yolo_out.append(out)
                #[batch,3,ny,nx,xywh+conf+cls]
                # box_idx=266, 267, 279, 280
                # print('yolo output:x[{},{},{},{}]:{}'.format(b,c,h,w,out[0,267,:4]))
            else:
                # print('conv:x input shape={}'.format(x.shape))
                b,c,h,w=0,2,4,5
                # print('conv input:x[{},{},{},{}]:{}'.format(b,c,h,w,x[b,c,h,w]))
                x = module(x)
                # print(module[0].weight.shape)
                # print(module[0].weight[31,1,1,1])
                #weight是正确的.可能是bn不对
                # print(module[1].weight.shape)
                # print(module[1].weight[4])
                # print('conv output:x[{},{},{},{}]:{}'.format(b,c,h,w,x[b,c,h,w]))
                # print('conv:x output shape={}'.format(x.shape))

            out_every_layer.append(x)

        return yolo_out
        
if __name__ == '__main__':
    yolov3net = Yolov3('cfg/yolov3_tlr.cfg')
    yolov3net.eval()
    # print(yolov3net.module_defs)
    for i, (mdef, module) in enumerate(zip(yolov3net.module_defs, yolov3net.module_list)):
        print(mdef,module)

    input = torch.randn((1,3,416,416))
    yolo_out=yolov3net(input)
    print([out.shape for out in yolo_out])

    
