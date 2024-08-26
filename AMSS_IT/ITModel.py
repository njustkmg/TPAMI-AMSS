import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset
from fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion,CMML
from pytorch_pretrained_bert.modeling import BertModel
import torchvision

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.fea_fc = nn.Linear(768, 256)
    def forward(self, txt):
        _, out = self.bert(
            txt,
            output_all_encoded_layers=False,
        )
        out = self.fea_fc(out)
        return out

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        # model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # modules = list(model.children())[:-1]
        # self.model = nn.Sequential(*modules)
        self.model = model
        num_features = self.model.fc.in_features
        self.model.fc =  nn.Linear(num_features, 256)
        
            # nn.Sigmoid()

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.model(x)
        return out # BxNx2048   

class ResnetClsModel(nn.Module):
    def __init__(self, config):
        super(ResnetClsModel, self).__init__()

        self.imgenc = ImageEncoder(config)
        self.num_class=config.n_classes
        
        # self.out_fc = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(768, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.num_class),
        # )
        self.out_fc =  nn.Linear(256, self.num_class)
    def forward(self, img):
        img_vec = self.imgenc(img)
        feature = img_vec
        out = self.out_fc(feature)
        return feature, out

class BertClsModel(nn.Module):

    def __init__(self, config):
        super(BertClsModel, self).__init__()
        self.txtenc = BertEncoder(config)
        self.num_class=config.n_classes
        # self.drop = nn.Dropout(p=0.1)
        self.out_fc = nn.Linear(256, self.num_class)
        
        # self.out_fc =  nn.Linear(768, self.num_class)
    def forward(self, text):
        text_vec = self.txtenc(text)
        out = self.out_fc(text_vec)
        feature = text_vec
        return feature, out
  
class JointClsModel(nn.Module):
    def __init__(self, config):
        super(JointClsModel, self).__init__()
        self.num_class=config.n_classes
        self.fusion_method  = config.fusion_method
        self.textenc = BertEncoder(config)
        self.imgenc = ImageEncoder(config)
        self.hidden_dim = 256
        if self.fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=self.hidden_dim,output_dim=config.n_classes)
        elif self.fusion_method == 'concat' or self.fusion_method == 'ORG_GE':
            self.fusion_module = ConcatFusion(input_dim=self.hidden_dim*2,output_dim=config.n_classes)
        elif self.fusion_method == 'film':
            self.fusion_module = FiLM(input_dim=self.hidden_dim,dim=self.hidden_dim,output_dim=config.n_classes, x_film=True)
        elif self.fusion_method == 'gated':
            self.fusion_module = GatedFusion(input_dim=self.hidden_dim,output_dim=config.n_classes, x_gate=True)
        elif self.fusion_method == 'cmml':
            self.fusion_module = CMML(input_dim=self.hidden_dim,output_dim=config.n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion_method))
            
    def forward(self, text, img):

        img_feat = self.imgenc(img)
        text_feat = self.textenc(text)
        # img_feat = self.img_fc(img_feat)
        # text_feat = self.textenc(text_feat)

        # output = torch.cat((img_feat, text_feat), dim=1)
        _,_,out = self.fusion_module(text_feat,img_feat)
        # feature = self.view_1_view_0_cat_fc(output)
        # out = self.fc_out(output)
        return text_feat,img_feat, out 


