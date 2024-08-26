import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import resnet18
from fusion_modules import SumFusion, ConcatFusion, FiLM,CMML
# import torchvision.models as models
# import torchaudio.models.rnnt

class AudioEncoder(nn.Module):
    def __init__(self,mask_model=1):
        super(AudioEncoder, self).__init__()
        self.mask_model=mask_model
        self.audio_net = resnet18(modality='audio')        
    def forward(self, audio):
    
        a = self.audio_net(audio) 
        a = F.adaptive_avg_pool2d(a, 1) #[512,1]
        a = torch.flatten(a, 1) #[512]
        return a
    
class VideoEncoder(nn.Module):
    def __init__(self,fps,mask_model=1):
        super(VideoEncoder, self).__init__()
        self.mask_model = mask_model
        self.video_net = resnet18(modality='visual')
        self.fps = fps

    def forward(self, video):

        v  = self.video_net(video)
        (_, C, H, W) = v.size()
        B = int(v.size()[0]/self.fps)
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        v = torch.flatten(v, 1)  
        return v
    
class AudioClassifier(nn.Module):
    def __init__(self,n_classes):
        super(AudioClassifier, self).__init__()
        self.audio_net = AudioEncoder()
        self.out_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(512), #-----------添加
            nn.ReLU(),#-----------添加
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes),
            # nn.Sigmoid()
        )

    def forward(self, audio):
        a_feature= self.audio_net(audio,balance=0)
        out = self.classifier(a_feature)
        return a_feature,out 
    
class VideoClassifier(nn.Module):
    def __init__(self,n_classes,fps):   
        super(VideoClassifier, self).__init__()
        self.video_net = VideoEncoder(fps)
        self.out_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(512), #-----------添加
            nn.ReLU(),#-----------添加
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_classes),
            # nn.Sigmoid()
        )
    def forward(self, video):
        v_feature= self.video_net(video,balance=0)
        out = self.classifier(v_feature)
        return v_feature,out 

class AVClassifier(nn.Module):
    def __init__(self, config,mask_model=1, act_fun=nn.GELU()):
        super(AVClassifier, self).__init__()
        self.fusion = config.fusion_method
        self.mask_ffn = config.mask_ffn
        self.mask_model=mask_model
        self.audio_net = AudioEncoder(mask_model)
        self.video_net = VideoEncoder(config.fps,mask_model)
        self.hidden_dim = 512
        if self.fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=config.n_classes)
        elif self.fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim=self.hidden_dim*2,output_dim=config.n_classes)
        elif self.fusion == 'film':
            self.fusion_module = FiLM(output_dim=config.n_classes, x_film=True)
        elif self.fusion == 'cmml':
            self.fusion_module = CMML(output_dim=config.n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(self.fusion))
        

    def forward(self, audio, video,step=0,balance_a=0,balance_v=0,s=400,a_bias=0,v_bias=0):
        #balance_modal=1表示平衡模态1 
        # if balance_modal==2:
        #     print('warmup test') 
        #     a_feature,audio_keep_list = self.audio_net(audio,False,s)
        #     v_feature,visual_keep_list = self.video_net(video,False,s)
        # print(audio.shape, video.shape)
        if self.training == True:
            a_feature = self.audio_net(audio)
            v_feature = self.video_net(video)
        else:
            a_feature = self.audio_net(audio)
            v_feature = self.video_net(video)
            
        _,_,out = self.fusion_module(a_feature, v_feature)

        return a_feature,v_feature,out

