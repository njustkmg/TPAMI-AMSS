import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import copy
import math
class CMML(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(CMML, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, y):
        audio_attention = self.softmax(x.clone())
        visual_attention = self.softmax(y.clone())
        sum_modal =  audio_attention+visual_attention
        audio_attention = audio_attention/sum_modal
        visual_attention = visual_attention/sum_modal
        supervise_feature_hidden = audio_attention * x + visual_attention * y
        output = self.fc_out(supervise_feature_hidden)
        return x, y, output
    
class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output
      
class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


    def __init__(self, input_dim=1024, output_dim=31):
        super(MaskConcatFusion, self).__init__()

        self.in_features = input_dim
        self.out_features = output_dim
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.test_mask = None
        # print(self.fc_out.weight.size(1)/2,self.fc_out.weight.size(0))
        self.efc_x = nn.Embedding(int(self.fc_out.weight.size(1)/2),self.fc_out.weight.size(0))
        self.efc_y = nn.Embedding(int(self.fc_out.weight.size(1)/2),self.fc_out.weight.size(0))
        self.efc_x.weight.requires_grad = False
        self.efc_y.weight.requires_grad = False
        
        self.gate = torch.nn.Sigmoid()
        
    def forward(self, x, y,s=0,bias_x=0,bias_y=0,mask_ffn=False):
        input = torch.cat((x, y), dim=1)
        if self.training and bias_x and bias_y and mask_ffn :
            mask_x = torch.transpose(self.mask(0,s=s,bias=bias_x),0,1)
            mask_y = torch.transpose(self.mask(1,s=s,bias=bias_y),0,1)
            adaptor_thresholded = torch.cat([mask_x, mask_y], dim=1)
            weight_thresholded = adaptor_thresholded * self.fc_out.weight
            return None,adaptor_thresholded,F.linear(input, weight_thresholded, self.fc_out.bias)
        
        else:
            weight_thresholded = self.fc_out.weight
            return None,None,F.linear(input, weight_thresholded, self.fc_out.bias)

    def mask(self,t,s,bias):
        if t==0:   
            el = self.efc_x.weight
        else:
            el = self.efc_y.weight
            
        sort_el = torch.sort(copy.deepcopy(el), dim=1)[0]
        bias_indices = math.ceil(sort_el.size(1) * bias) - 1
        bias_e = sort_el[:,bias_indices]  # 获取每行对应的偏置值
        # bias_e = sort_el[math.ceil(len(sort_el)*bias)].item()
        return self.gate(s * (el + bias_e.unsqueeze(1)))