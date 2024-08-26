import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import resnet18

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
        a_feature,_ = self.audio_net(audio,balance=0)
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
        v_feature,_ = self.video_net(video,balance=0)
        out = self.classifier(v_feature)
        return v_feature,out 
    
class MMTM(nn.Module):
    def __init__(self, 
            dim_visual, 
            dim_skeleton, 
            ratio, 
            SEonly=False, 
            shareweight=False):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2*dim/ratio)
        self.SEonly = SEonly
        self.shareweight = shareweight
        self.running_avg_weight_visual = torch.zeros(dim_visual).cuda()
        self.running_avg_weight_skeleton = torch.zeros(dim_visual).cuda()
        self.step = 0
        if self.SEonly:
            self.fc_squeeze_visual = nn.Linear(dim_visual, dim_out)
            self.fc_squeeze_skeleton = nn.Linear(dim_skeleton, dim_out)
        else:    
            self.fc_squeeze = nn.Linear(dim, dim_out)

        if self.shareweight:
            assert dim_visual == dim_skeleton
            self.fc_excite = nn.Linear(dim_out, dim_visual)
        else:
            self.fc_visual = nn.Linear(dim_out, dim_visual)
            self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, 
                visual, 
                skeleton, 
                return_scale=False,
                return_squeezed_mps = False,
                turnoff_cross_modal_flow = False,
                average_squeezemaps = None,
                curation_mode=False,
                caring_modality=0,
                ):

        if self.SEonly:
            tview = visual.view(visual.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_visual(squeeze)
            vis_out = self.fc_visual(self.relu(excitation))

            tview = skeleton.view(skeleton.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_skeleton(squeeze)
            sk_out = self.fc_skeleton(self.relu(excitation))

        else:
            if turnoff_cross_modal_flow: #一般情况
                tview = visual.view(visual.shape[:2] + (-1,))
                squeeze = torch.cat([torch.mean(tview, dim=-1), 
                    torch.stack(visual.shape[0]*[average_squeezemaps[1]])], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))
                if self.shareweight:
                    vis_out = self.fc_excite(excitation)
                else:
                    vis_out = self.fc_visual(excitation)

                tview = skeleton.view(skeleton.shape[:2] + (-1,))
                squeeze = torch.cat([torch.stack(skeleton.shape[0]*[average_squeezemaps[0]]), 
                    torch.mean(tview, dim=-1)], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))
                if self.shareweight:
                    sk_out = self.fc_excite(excitation)
                else:
                    sk_out = self.fc_skeleton(excitation)

            else:  #文章网络为该网络----------------------------------------------------------------------------------------------------
                squeeze_array = []
                for tensor in [visual, skeleton]:
                    tview = tensor.view(tensor.shape[:2] + (-1,))
                    squeeze_array.append(torch.mean(tview, dim=-1))
                
                squeeze = torch.cat(squeeze_array, 1)
                excitation = self.fc_squeeze(squeeze)
                excitation = self.relu(excitation)

                if self.shareweight:
                    sk_out = self.fc_excite(excitation)
                    vis_out = self.fc_excite(excitation)
                else:
                    vis_out = self.fc_visual(excitation)
                    sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        self.running_avg_weight_visual = (vis_out.mean(0) + self.running_avg_weight_visual*self.step).detach()/(self.step+1)
        self.running_avg_weight_skeleton = (vis_out.mean(0) + self.running_avg_weight_skeleton*self.step).detach()/(self.step+1)
        self.step +=1

        if return_scale:
            scales = [vis_out.cpu(), sk_out.cpu()]
        else:
            scales = None

        if return_squeezed_mps:
            squeeze_array = [x.cpu() for x in squeeze_array]
        else:
            squeeze_array = None

        if not curation_mode:
            dim_diff = len(visual.shape) - len(vis_out.shape)
            vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

            dim_diff = len(skeleton.shape) - len(sk_out.shape)
            sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)
        
        else:
            if caring_modality==0:
                dim_diff = len(skeleton.shape) - len(sk_out.shape)
                sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

                dim_diff = len(visual.shape) - len(vis_out.shape)
                vis_out = torch.stack(vis_out.shape[0]*[
                        self.running_avg_weight_visual
                    ]).view(vis_out.shape + (1,) * dim_diff)
                
            elif caring_modality==1:
                dim_diff = len(visual.shape) - len(vis_out.shape)
                vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

                dim_diff = len(skeleton.shape) - len(sk_out.shape)
                sk_out = torch.stack(sk_out.shape[0]*[
                        self.running_avg_weight_skeleton
                    ]).view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out, scales, squeeze_array

class MMTMClassifier(nn.Module):
    def __init__(self, 
        config, 
        num_views=2, 
        saving_mmtm_scales=False,
        saving_mmtm_squeeze_array=False,
        ):
        super(MMTMClassifier, self).__init__()
        self.mmtm_off = False
        self.nclasses = config.n_classes
        self.num_views = num_views
        self.saving_mmtm_scales = saving_mmtm_scales
        self.saving_mmtm_squeeze_array = saving_mmtm_squeeze_array
        self.audio_net = resnet18(modality='audio')
        self.auido_fc = nn.Linear(512, config.n_classes)
        self.video_net = resnet18(modality='visual')
        self.video_fc = nn.Linear(512, config.n_classes)
        if config.dataset=='KS':
            self.fps = 3
        elif config.dataset=='AVE':
            self.fps = 4
        else:
            self.fps = 1
        self.mmtm2 = MMTM(128, 128, 4)
        self.mmtm3 = MMTM(256, 256, 4)
        self.mmtm4 = MMTM(512, 512, 4)


    def forward(self, audio, video, curation_mode=False, caring_modality=None):
        #[64, 3, 7, 7]
        '''     
        # torch.Size([8, 1, 257, 188]) audio
        # torch.Size([8, 3, 224, 224]) visual 
        '''
        # print(x[0].shape)
        # print(x[1].shape)

        # frames_audio = self.audio_net.conv1(x[[:, 0, :]])
        frames_audio = self.audio_net.conv1(audio)
        frames_audio = self.audio_net.bn1(frames_audio)
        frames_audio = self.audio_net.relu(frames_audio)
        frames_audio = self.audio_net.maxpool(frames_audio)

        (B, C, T, H, W) = video.size()
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        video = video.view(B * T, C, H, W)
        # frames_video = self.video_net.conv1(x[:, 1, :])

        frames_video = self.video_net.conv1(video)
        frames_video = self.video_net.bn1(frames_video)
        frames_video = self.video_net.relu(frames_video)
        frames_video = self.video_net.maxpool(frames_video)

        frames_audio = self.audio_net.layer1(frames_audio) 
        frames_video = self.video_net.layer1(frames_video) 

        scales = []
        squeezed_mps = [] 

        for i in [2, 3, 4]:

            frames_audio = getattr(self.audio_net, f'layer{i}')(frames_audio)
            frames_video = getattr(self.video_net, f'layer{i}')(frames_video)

            frames_audio, frames_video, scale, squeezed_mp = getattr(self, f'mmtm{i}')(
                frames_audio, 
                frames_video, 
                self.saving_mmtm_scales,
                self.saving_mmtm_squeeze_array,
                turnoff_cross_modal_flow =False,
                average_squeezemaps =  None,
                curation_mode = curation_mode,
                caring_modality = caring_modality
                )
            scales.append(scale)
            squeezed_mps.append(squeezed_mp)

        frames_audio = F.adaptive_avg_pool2d(frames_audio,1)# C维向量
        frames_video = F.adaptive_avg_pool2d(frames_video,1)# C维向量

        (_, C, H, W) = frames_video.size()
        B = int(frames_video.size()[0])
        frames_video = frames_video.view(B, -1, C, H, W)
        frames_video = frames_video.permute(0, 2, 1, 3, 4)
        frames_video = F.adaptive_avg_pool3d(frames_video, 1)
        
        x0_feature = torch.flatten(frames_audio, 1)
        x_0 = self.auido_fc(x0_feature)
        
        x1_feature = torch.flatten(frames_video, 1)
        x_1 = self.video_fc(x1_feature)
        
        return x0_feature,x1_feature,(x_0+x_1)/2
        # return a_feature,v_feature,out

