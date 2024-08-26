import torch.nn as nn
import torch
from fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion,CMML
from backbone import InceptionI3d

class RGBEncoder(nn.Module):
    def __init__(self,config):
        super(RGBEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=3)
        if config.single_pretrain==1:
            pretrained_dict  = torch.load('models/rgb_imagenet.pt')
            model_dict = model.state_dict()
            # 过滤掉不匹配的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新新模型的权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        self.rgbmodel = model

    def forward(self, x):
        out = self.rgbmodel(x)
        return out # BxNx2048 

class OFEncoder(nn.Module):
    def __init__(self,config):
        super(OFEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=2)
        if config.single_pretrain==1:
            pretrained_dict  = torch.load('models/flow_imagenet.pt')
            model_dict = model.state_dict()
            # 过滤掉不匹配的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新新模型的权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.ofmodel = model
        
    def forward(self, x):
        out = self.ofmodel(x)
        return out # BxNx2048  
      
class DepthEncoder(nn.Module):
    def __init__(self,config):
        super(DepthEncoder, self).__init__()
        model = InceptionI3d(400, in_channels=1)
        if config.single_pretrain==1:
            pretrained_dict  = torch.load('models/rgb_imagenet.pt')
            model_dict = model.state_dict()
            # 过滤掉不匹配的权重
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 更新新模型的权重
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.depthmodel = model
        
    def forward(self, x):
        out = self.depthmodel(x)
        return out # BxNx2048 

class RGBClassifier(nn.Module):
    def __init__(self,config):
        super(RGBClassifier, self).__init__()
        self.rgbnet = RGBEncoder(config)
        self.classifier = nn.Linear(1024, config.n_classes)
    def forward(self, modal):
        feature= self.rgbnet(modal)
        out = self.classifier(feature)
        return out 

class OFClassifier(nn.Module):
    def __init__(self,config):
        super(OFClassifier, self).__init__()
        self.of_net = OFEncoder(config)
        self.classifier = nn.Linear(1024, config.n_classes)

    def forward(self, of):
        of_feature= self.of_net(of)
        out = self.classifier(of_feature)
        return out 

class DEPTHClassifier(nn.Module):
    def __init__(self,config):
        super(DEPTHClassifier, self).__init__()
        self.depth_net = DepthEncoder(config)
        self.classifier = nn.Linear(1024, config.n_classes)

    def forward(self, depth):
        depth_feature= self.depth_net(depth)
        out = self.classifier(depth_feature)
        return out 

class JointClsModel(nn.Module):
    def __init__(self, config):
        super(JointClsModel, self).__init__()
        self.num_class=config.n_classes
        self.fusion_method  = None
        self.rgbenc = RGBEncoder(config)
        self.ofenc = OFEncoder(config)
        self.depthenc = DepthEncoder(config)
        self.hidden_dim = 1024
        
        if config.fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=self.hidden_dim,output_dim=config.n_classes)
        elif config.fusion_method == 'concat':
            self.fusion_module = ConcatFusion(input_dim=self.hidden_dim*3,output_dim=config.n_classes)
        elif config.fusion_method == 'film':
            self.fusion_module = FiLM(input_dim=self.hidden_dim,dim=self.hidden_dim,output_dim=config.n_classes, x_film=True)
        elif config.fusion_method == 'gated':
            self.fusion_module = GatedFusion(input_dim=self.hidden_dim,output_dim=config.n_classes, x_gate=True)
        elif config.fusion_method == 'cmml':
            self.fusion_module = CMML(input_dim=self.hidden_dim,output_dim=config.n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion_method))
            
    def forward(self, rgb, of, depth, mode = 0 ):

        rgb_feat = self.rgbenc(rgb)
        of_feat = self.ofenc(of)
        depth_feat = self.depthenc(depth)
        if mode == 0:
            out = self.fusion_module(rgb_feat, of_feat, depth_feat, 0)
            return  rgb_feat, of_feat, depth_feat, out
        else:
            rgb_out,of_out,depth_out = self.fusion_module(rgb_feat,of_feat,depth_feat,1)
            return rgb_out,of_out,depth_out
    
