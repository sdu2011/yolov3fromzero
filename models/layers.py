import torch
import torch.nn as nn

class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers

    def forward(self,feature_list):
        """
        out:记录了各个layer的输出
        """
        for i,feature in enumerate(feature_list):
            # print('feature{} shape:{}'.format(i,feature.shape))
            pass
        out = torch.cat(feature_list,1)
        # print('out shape{}'.format(out.shape))

        return out

class FeatureShortcut(nn.Module):
    def __init__(self, from_layers):
        super(FeatureShortcut, self).__init__()
        self.from_layers = from_layers

    def forward(self,x,y):
        # print('x shape={},y shape={}'.format(x.shape,y.shape))
        return x + y

class YoloLayer(nn.Module):
    def __init__(self, input_img_size,anchors,cls_num,stride,yolo_index=1):
        """
        img_size:(h,w)
        """
        super(YoloLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.cls_num = cls_num
        self.stride = stride 
        self.anchors_num = len(self.anchors)
        self.index = yolo_index

        self.anchor_scale = self.anchors/self.stride
        #配置文件中的anchor是相对于原图的,这里做了下采样.
        self.anchor_wh = self.anchor_scale.view(1,len(anchors),1,1,2)
        #shape匹配yolo层的输出(bs, anchors, grid, grid, xywh+conf+cls)

        # self.ny,self.nx = -1,-1
        self.ny,self.nx = input_img_size[0]//stride,input_img_size[1]//stride
        self.grid = self.__make_grid(self.nx,self.ny)
        # print('nx={}'.format(self.nx))


        # print('layer{},anchors:{},nx:{},anchor_scale:{}'.\
        #     format(yolo_index,self.anchors,nx,self.anchor_scale))

    def get_grid(self):
        return self.nx,self.ny

    def forward(self,x):
        """
        x:[batch,3x(5+cls),ny,nx]    
        """
        # print('x shape={}'.format(x.shape))
        batch,_,ny,nx = x.shape
        x = x.view(batch,self.anchors_num,(5+self.cls_num),ny,nx).permute(0,1,3,4,2).contiguous()
        #[batch,3x(5+cls),n,n] --> [batch,anchors,gridy,gridx,5+cls]
        if self.training:
            return x
        else:
            # inference
            # print('yolo layer{} inference'.format(self.index))            
            model_out = x.clone()
            self.decode(model_out)
            model_out[...,0:4] = model_out[...,0:4] * self.stride
            #还原到在原图的位置(模型输入的图片)

            bs,_,_,_,n = model_out.shape
            model_out = model_out.view(bs,-1,n)

            # mask = (model_out[...,4] > 0.9)
            # print('满足条件的输出为:')
            # print(model_out[mask])

            return model_out
            
    def __make_grid(self,nx=13,ny=13):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid = torch.stack((xv,yv),2)
        
        grid = grid.view((1,1,ny,nx,2))

        return grid

    def set_gridxy(self,gridx,gridy):
        self.nx,self.ny = gridx,gridy

    def decode(self,model_out):
        """
        model_out: [batch,3,ny,nx,xywh+conf+cls]

        return:在当前feature map上的x,y,w,h
        """
        _,_,ny,nx,_ = model_out.shape
        if (self.nx,self.ny) != (nx,ny):
            print('nx={},ny={}'.format(nx,ny))
            self.grid = self.__make_grid(nx,ny)
            #make_grid只做一次.
        #解决multi-scale训练的问题. grid的大小不能在初始化时完成.

        if model_out.is_cuda and not self.grid.is_cuda:
            self.grid = self.grid.cuda()
            self.anchor_wh = self.anchor_wh.cuda()

        # x=torch.sigmoid(model_out[...,0])
        # print('x={}'.format(x))

        model_out[...,0:2] = torch.sigmoid(model_out[...,0:2]) + self.grid
        model_out[...,2:4] = torch.exp(model_out[...,2:4]) * self.anchor_wh
        model_out[...,4:] = torch.sigmoid(model_out[..., 4:])

        return model_out




