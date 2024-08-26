# -*- coding: utf-8 -*-
import argparse
import csv
import math
import shutil
import sys
import json
import time
# from torchnet import meter
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from RODModel import JointClsModel
# from MMTM  import MMTMClassifier
from MyHelper import get_ROD_data_loaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,average_precision_score
import os
import random
import matplotlib.pyplot as plt
# from thop import profile
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
sys.path.append('../..')

'''
强弱模态量化代码
'''
def entropy(probabilities, epsilon=1e-10):
    # 将概率转换为对数概率，添加平滑项
    log_probabilities = torch.log2(probabilities + epsilon)
    # 计算信息熵
    entropy = -torch.sum(probabilities * log_probabilities)
    return entropy

def limit(num, bias):
    return torch.clamp(num-bias,0.1,0.9)
def modal_caculate_MI(r_out,o_out,d_out,r,o,d,labels):
    def limit_number(num):
        if torch.isnan(num):
            return torch.tensor(0.5)
        return torch.clamp(num,0.1,0.9)
    label_single = torch.argmax(labels,1)
    label_counts = torch.bincount(label_single)
    label_probs = label_counts.float() / len(label_single)
    # 计算标签熵
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    HY = entropy(label_probs)
    # HY =  torch.sum(-label_probs * torch.log2(label_probs))
    # softmax = nn.Softmax(dim=1)

    T = config.temperature
    score_r = -criterion(r_out,label_single)
    score_o = -criterion(o_out,label_single)
    score_d = -criterion(d_out,label_single)

    # score_a = sum([torch.log(softmax(a_out)[i][torch.argmax(labels[i]).item()]) for i in range(a_out.size(0))])/a_out.size(0)
    pr = softmax(r)
    po = softmax(o)
    pd = softmax(d)

    Hr = entropy(pr)/40
    Ho = entropy(po)/40
    Hd = entropy(pd)/40
    # print(HY.item(),Hr.item(),Ho.item(),Hd.item())
    if  (score_r+HY+2)<=0 or (score_o+HY+2)<=0 or (score_d+HY+2)<0:
        coeff_r,coeff_o,coeff_d = torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5)
        ratio_r,ratio_o,ratio_d = torch.tensor(1), torch.tensor(1), torch.tensor(1)
        t_r,t_o,t_d = 1,1,1
    else:
        ratio_r = (score_r+HY+3)/Hr
        ratio_o = (score_o+HY+3)/Ho
        ratio_d = (score_d+HY+3)/Hd
        # ratio_v =  ratio_a_u/ratio_v_u  # 反应visual模态与rgb模态的强弱比 ；弱化强模态
        # ratio_a =  1 / ratio_v
        t_r, t_o, t_d = ratio_r/T, ratio_o/T, ratio_d/T

        coeff_r = 1-torch.exp(t_r)/(torch.exp(t_r)+torch.exp(t_o)+torch.exp(t_d))+config.bias
        coeff_o = 1-torch.exp(t_o)/(torch.exp(t_r)+torch.exp(t_o)+torch.exp(t_d))+config.bias
        coeff_d = 1-torch.exp(t_d)/(torch.exp(t_r)+torch.exp(t_o)+torch.exp(t_d))+config.bias

        coeff_r = limit_number(coeff_r)
        coeff_o = limit_number(coeff_o)
        coeff_d = limit_number(coeff_d)

    return coeff_r,coeff_o,coeff_d

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)    
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(config, model, criterion, optimizer, train_loader,scheduler,epoch):

    model.train()
    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []
    num_rod, acc_r, acc_o, acc_d = [0.0] * config.n_classes, [0.0] * config.n_classes, [0.0] * config.n_classes, [0.0] * config.n_classes

    train_tqdm = tqdm(train_loader)
    softmax = nn.Softmax(dim=1)
    hidden_dim = 1024*3
    for index,batch in enumerate(train_tqdm):
        optimizer.zero_grad()
        rgb, of, depth, label = batch
        rgb, of, depth, label = rgb.cuda(), of.cuda().float(), depth.cuda(), label.cuda()
        # print(spectrogram.unsqueeze(1).float().shape,images.float().shape)
        r,o,d,out = model(rgb, of, depth)

        if config.fusion_method == 'sum' :
            r_out = (torch.mm(r, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) +
                    model.module.fusion_module.fc_x.bias)
            o_out = (torch.mm(o, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                    model.module.fusion_module.fc_y.bias)
            d_out = (torch.mm(d, torch.transpose(model.module.fusion_module.fc_z.weight, 0, 1)) +
                    model.module.fusion_module.fc_z.bias)
        elif config.fusion_method == 'concat':
            r_out = (torch.mm(r, torch.transpose(model.fusion_module.fc_out.weight[:, :hidden_dim//3], 0, 1)) +
                    model.fusion_module.fc_out.bias / 3)
            o_out = (torch.mm(o, torch.transpose(model.fusion_module.fc_out.weight[:, hidden_dim//3:(hidden_dim//3)*2], 0, 1)) +
                    model.fusion_module.fc_out.bias / 3)
            d_out = (torch.mm(d, torch.transpose(model.fusion_module.fc_out.weight[:, (hidden_dim//3)*2:], 0, 1)) +
                    model.fusion_module.fc_out.bias / 3)
        else:
            r_out,o_out,d_out = out,out,out
        coeff_r,coeff_o,coeff_d = modal_caculate_MI(r_out,o_out,d_out,r,o,d,label.cuda())
        
        if config.fusion_method != 'sum' and config.fusion_method != 'concat' and config.fusion_method != 'mmtm':
            coeff_r = 0.33 
            coeff_o = 0.33
            coeff_d = 0.33

        label_single = torch.argmax(label,1)
        pred_r,pred_o,pred_d = softmax(r_out),softmax(o_out),softmax(d_out)

        for i in range(rgb.shape[0]):
            num_rod[label_single[i].cpu()] += 1.0
            r_i = np.argmax(pred_r[i].cpu().data.numpy())
            o_i = np.argmax(pred_o[i].cpu().data.numpy())  
            d_i = np.argmax(pred_d[i].cpu().data.numpy())  

            if np.asarray(label_single[i].cpu()) == r_i:
                acc_r[label_single[i].cpu()] += 1.0
            if np.asarray(label_single[i].cpu()) == o_i:
                acc_o[label_single[i].cpu()] += 1.0
            if np.asarray(label_single[i].cpu()) == d_i:
                acc_d[label_single[i].cpu()] += 1.0
        
        if max(coeff_r,coeff_o,coeff_d) == coeff_r: modal_rgb[epoch]+=1
        elif max(coeff_r,coeff_o,coeff_d) == coeff_o: modal_of[epoch]+=1
        else: modal_depth[epoch]+=1

        if torch.is_tensor(coeff_r): coeff_r = coeff_r.item()
        if torch.is_tensor(coeff_o):  coeff_o = coeff_o.item()
        if torch.is_tensor(coeff_d):  coeff_d = coeff_d.item()

        # if config.ratio_random==1:
        #     coeff_r,coeff_o,coeff_d = random.random(),random.random(),random.random()
        #标签损失 + Mask损失计算
        loss = criterion(out,label_single.cuda())
        loss.backward()


        for name, parms in model.named_parameters():
            print(name, parms.shape)

            if config.mask_ffn and('fc_out.weight' in name):
                gn = parms.grad.data ** 2
                mask_miu = (gn).cuda()
                gn_rgb, gn_of, gn_depth = (mask_miu)[:, :hidden_dim // 3], (mask_miu)[:, hidden_dim // 3 : (hidden_dim // 3) * 2], (mask_miu)[:, (hidden_dim // 3) * 2 : ]

                p_r, p_o, p_d = int(gn_rgb.size(1)*coeff_r), int(gn_of.size(1)*coeff_o), int(gn_of.size(1)*coeff_d)

                if config.sample_mode == 'GN' or config.sample_mode == 'Bernoulli':

                    rgb_fcn_indices, of_fcn_indices, depth_fcn_indices = torch.argsort(gn_rgb)[:,-p_r:], torch.argsort(gn_of)[:,-p_o:], torch.argsort(gn_depth)[:,-p_d:]

                elif config.sample_mode == 'random':
                    # # 定义需要计算的变量列表
                    variables = [("gn_rgb", p_r, "rgb_fcn_indices"), ("gn_of", p_o, "of_fcn_indices"),("gn_depth", p_d, "depth_fcn_indices")]

                    for var_name, p, index_name in variables:
                        data = np.zeros((vars()[var_name].shape[0], p), dtype=int)
                        for i in range(vars()[var_name].shape[0]):
                            data[i] = np.random.choice(range(vars()[var_name].shape[1]), size=p, replace=False)
                        vars()[index_name] = torch.from_numpy(data)

                elif config.sample_mode == 'Adaptive':

                    rgb_importance, of_importance, depth_importance= gn_rgb / gn_rgb.sum(), gn_of / gn_of.sum(), gn_depth / gn_depth.sum()
                         
                    variables = [("gn_rgb", rgb_importance, p_r, "rgb_fcn_indices"), ("gn_of", of_importance, p_o, "of_fcn_indices"), ("gn_depth", depth_importance, p_d, "depth_fcn_indices")]

                    for var_name, importance, p, index_name in variables:
                        data = torch.zeros((vars()[var_name].shape[0], p), dtype=int)
                        for i in range(vars()[var_name].shape[0]):
                            data[i] = torch.multinomial(importance[i], num_samples=p, replacement=False)
                        vars()[index_name] = data
                        
                mask_r_fcn, mask_o_fcn, mask_d_fcn = torch.zeros_like(gn_rgb), torch.zeros_like(gn_of), torch.zeros_like(gn_depth)

                variables = [("gn_rgb", rgb_fcn_indices, mask_r_fcn), ("gn_of", of_fcn_indices, mask_o_fcn), ("gn_depth", depth_fcn_indices, mask_d_fcn)]
                for var_name, indices, mask_var in variables:
                    for i, slice in enumerate(indices):
                        if config.isbias == 1:
                            mask_var[i][slice] = (1 / (slice + 1 / len(slice))).cuda()
                        else:
                            mask_var[i][slice] = 1

                if config.sample_mode != 'Bernoulli':
                    parms.grad[:, :hidden_dim // 3] *= mask_r_fcn
                    parms.grad[:, hidden_dim // 3 : (hidden_dim // 3) * 2] *= mask_o_fcn
                    parms.grad[:, (hidden_dim // 3) * 2 :] *= mask_d_fcn
            
        for module_name in ['rgbmodel', 'ofmodel', 'depthmodel']:

            if config.mask_resnet and ('conv3d.weight' in name) and (module_name in name):
                print(name, parms.shape)
                gn = torch.sum(parms.grad.data ** 2, dim=(1, 2, 3, 4))
                mask_miu = gn.cuda()

                if module_name == 'rgbmodel':
                    p = math.ceil(len(mask_miu) * coeff_r)
                elif module_name == 'ofmodel':
                    p = math.ceil(len(mask_miu) * coeff_o)
                elif module_name == 'depthmodel':
                    p = math.ceil(len(mask_miu) * coeff_d)

                if config.sample_mode == 'TopK':
                    max_p_indices = torch.argsort(mask_miu)[-p:]
                elif config.sample_mode == 'random':
                    max_p_indices = torch.randperm(len(mask_miu))[:p]
                elif config.sample_mode == 'Bernoulli':
                    important_gn = mask_miu/sum(mask_miu)
                    max_p_indice = torch.bernoulli(important_gn)
                    max_p_indices  = torch.nonzero(max_p_indice).squeeze()
                elif config.sample_mode == 'Adaptive':
                    important_gn = mask_miu / mask_miu.sum()
                    max_p_indices = torch.multinomial(important_gn, num_samples=p, replacement=False)

                mask = torch.zeros_like(mask_miu)
                if config.isbias == 1 and config.sample_mode == 'Adaptive':
                    mask[max_p_indices] = 1 / (important_gn[max_p_indices] + 1 / len(important_gn))
                else:
                    mask[max_p_indices] = 1
                parms.grad *= mask.unsqueeze(dim=(1, 2, 3, 4)).expand_as(parms.grad)

        optimizer.step()
        temp_out = out.detach().cpu()
        preds = torch.argmax(temp_out, 1)
        all_score += preds.numpy().tolist()
        all_out += out.detach().cpu().numpy().tolist()
        all_label += label
        all_loss.append(loss.item())
        train_tqdm.set_description('Loss: %.3f, coeff_r: %.3f,coeff_o: %.3f,,ratio_d: %.3f' \
            % (np.mean(all_loss),coeff_r,coeff_o,coeff_d))
        all_label_single +=label_single.detach().cpu().tolist()
   
    if config.optimizer=='SGD':
        scheduler.step()
    acc = accuracy_score(all_label_single, all_score)*100
    return np.mean(all_loss), acc

def evaluate(config, model, criterion, valid_loader,epoch):

    model.eval()
    all_out = []
    all_score = []
    all_label = []
    all_loss = []

    all_label_single=[]
    softmax = nn.Softmax(dim=1)
    num_rod = [0.0 for _ in range(config.n_classes)]
    acc_r = [0.0 for _ in range(config.n_classes)]
    acc_o = [0.0 for _ in range(config.n_classes)]
    acc_d = [0.0 for _ in range(config.n_classes)]

    valid_len = len(valid_loader)
    valid_tqdm = tqdm(valid_loader)

    for index, batch in enumerate(valid_tqdm):
        with torch.no_grad():
            rgb, of, depth, label = batch
            # print(spectrogram.shape)
            rgb, of, depth, label = rgb.cuda(), of.cuda().float(), depth.cuda(), label.cuda()
            r,o,d,out = model(rgb, of, depth)
            if config.fusion_method == 'sum' :
                r_out = (torch.mm(r, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                        model.module.fusion_module.fc_x.bias)
                o_out = (torch.mm(o, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                        model.module.fusion_module.fc_y.bias)
                d_out = (torch.mm(d, torch.transpose(model.module.fusion_module.fc_z.weight, 0, 1)) +
                        model.module.fusion_module.fc_z.bias)
            elif config.fusion_method == 'concat':
                r_out = (torch.mm(r, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                        model.module.fusion_module.fc_out.bias / 3)
                o_out = (torch.mm(o, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:1024], 0, 1)) +
                        model.module.fusion_module.fc_y.bias)
                d_out = (torch.mm(d, torch.transpose(model.module.fusion_module.fc_out.weight[:, 1024:], 0, 1)) +
                        model.module.fusion_module.fc_z.bias)
            else:
                r_out,o_out,d_out = out,out,out
            label_single = torch.argmax(label,1)
            pred_r,pred_o,pred_d = softmax(r_out),softmax(o_out),softmax(d_out)

            for i in range(rgb.shape[0]):
                num_rod[label_single[i].cpu()] += 1.0
                r_i = np.argmax(pred_r[i].cpu().data.numpy())
                o_i = np.argmax(pred_o[i].cpu().data.numpy())  
                d_i = np.argmax(pred_d[i].cpu().data.numpy())  

                if np.asarray(label_single[i].cpu()) == r_i:
                    acc_r[label_single[i].cpu()] += 1.0
                if np.asarray(label_single[i].cpu()) == o_i:
                    acc_o[label_single[i].cpu()] += 1.0
                if np.asarray(label_single[i].cpu()) == d_i:
                    acc_d[label_single[i].cpu()] += 1.0
            loss = criterion(out, label_single.cuda())    
            all_loss.append(loss.item())
            tmp_out = out.detach().cpu()
            preds = torch.argmax(tmp_out, 1)
            predict = softmax(tmp_out)
        writer.add_scalars('test_loss',{'test_loss':loss.item()}, index+valid_len*epoch)
        all_out += predict.cpu().numpy().tolist()
        all_score += preds.numpy().tolist()
        all_label += label.detach().cpu().tolist()
        all_label_single +=label_single.detach().cpu().tolist()
        valid_tqdm.set_description('Loss: %.3f' \
        % (np.mean(all_loss)))

    auc = roc_auc_score(all_label, all_out,multi_class='ovo')*100
    acc = accuracy_score(all_label_single, all_score)*100
    AP = []
    all_label = np.array(all_label)
    all_out = np.array(all_out)
    for i in range(len(all_label[0])):
        AP.append(average_precision_score(all_label[:,i],all_out[:,i]))
    f1 = f1_score(all_label_single, all_score, average='macro')*100
    return {'loss': np.mean(all_loss),'rgb_acc':sum(acc_o) / sum(num_rod) * 100,'of_acc':sum(acc_o) / sum(num_rod) * 100,\
             'depth_acc':sum(acc_d) / sum(num_rod) * 100,\
             'acc':acc,'auc': auc,'f1':f1,'mAP':np.mean(AP)*100}

def train_test(config, model, train_loader, valid_loader):
    '''
    图形初始化
    '''
    if  config.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    elif config.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [config.patience, config.patience+config.p2] , gamma= 0.1)
    criterion = nn.CrossEntropyLoss()
    print('Best ACC设置为0')
    best_acc = 0

    result_path = 'results/AMSS'

    result_save_path = result_path +'/{}_seed{}.tsv'.format(config.our_model,str(config.random_seed))
    
    with open(result_save_path, 'a+', newline='', encoding='utf-8') as tsv_f:
        tsv_w = csv.writer(tsv_f, delimiter='\t')
        tsv_w.writerow(['epoch','train_loss','train_acc','test_loss','rgb_acc','of_acc','depth_acc','test_acc','test_auc','test_f1','test_mAP'])
        tsv_f.close()   
    
    for epoch in range(config.epoch):
        model.zero_grad()
        print('Is Train Epoch:{}'.format(epoch))
        train_loss, train_acc = \
            train_epoch(config, model, criterion, optimizer, train_loader,scheduler,epoch)
        if epoch >=20:
            valid_result = evaluate(config, model, criterion,valid_loader,epoch)
            writer.add_scalars('loss',{'test_loss':valid_result['loss']}, epoch+1)
            writer.add_scalars('Acc', {'test_acc':valid_result['acc']}, epoch+1)
            print('TrainACC:',train_acc,'Test Metircs:',valid_result['acc'],valid_result['auc'],valid_result['f1'],valid_result['mAP'])
            with open(result_save_path, 'a+', newline='', encoding='utf-8') as tsv_f:
                tsv_w = csv.writer(tsv_f, delimiter='\t')
                tsv_w.writerow([epoch,train_loss,train_acc,valid_result['loss'],valid_result['rgb_acc'],valid_result['of_acc'],valid_result['depth_acc'],valid_result['acc'],valid_result['auc'],valid_result['f1'],valid_result['mAP']])
                tsv_f.close() 
            
            if valid_result['acc']>best_acc:
                best_acc, best_auc, best_f1, best_mAP= valid_result['acc'],valid_result['auc'],valid_result['f1'],valid_result['mAP']
        else:
            with open(result_save_path, 'a+', newline='', encoding='utf-8') as tsv_f:
                tsv_w = csv.writer(tsv_f, delimiter='\t')
                tsv_w.writerow([epoch,train_loss,train_acc,'-','-','-','-','-','-','-','-'])
                tsv_f.close() 
        writer.add_scalars('loss',{'train_loss':train_loss}, epoch+1)
        writer.add_scalars('Acc', {'train_acc':train_acc}, epoch+1)
    model_path = config.model_path
    if valid_result['acc'] > best_acc:
        best_acc = valid_result['acc']
        torch.save(model, model_path)
    writer.close()
        
def run(config):

    set_seed(config.random_seed)
    config.n_gpu = torch.cuda.device_count()
    print("Num GPUs", config.n_gpu)
    print("Device", config.device)
    train_loader,test_loader = get_ROD_data_loaders(config)
  
    if config.dataset == 'nvGesture':
        config.n_classes = 25
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(config.dataset))

    model = JointClsModel(config)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()    

    print('==='*50)
    print('Begin Train {} Model'.format(config.our_model))
    print('GPU:{} => Dataset:{} => Classes:{} => Batchsize:{} => LearingR:{} => Epoch:{} => Mask_Resnet:{} => SampleMode {}'.\
          format(config.gpu_id,config.dataset,config.n_classes,config.batch_size,config.lr,config.epoch,config.mask_resnet,config.sample_mode))
    print('==='*50)       
    train_test(config, model, train_loader, test_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use gpu')
    parser.add_argument('--random_seed', type=int, default=42,help='random seed')
    parser.add_argument('--data_path', type=str, default='/media/php/data/nvGesture/')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--model_path', type=str, default='save_model_path')

    parser.add_argument('--epoch', type=int, default=50,
                        help='the total number of epoch in the training')
    parser.add_argument('--num_workers', type=int , default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--p2', type=int, default=10)

    parser.add_argument('--batch-size', default=2, type=int,help='train batchsize')
    parser.add_argument('--test-batch-size', default=4, type=int,help='train batchsize')
    parser.add_argument('--modal', type=str, default='DER_av_balance_v')
    parser.add_argument('--single_pretrain',type=int,default=1)
    parser.add_argument("--our_model", default="SEBM",type=str)
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--dataset',default='nvGesture',type=str)
    parser.add_argument('--mask_resnet', type=int, default=1)
    parser.add_argument('--mask_ffn', type=int, default=0)
    parser.add_argument('--isbias', type=int, default=0)
    parser.add_argument('--bias', type=float, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--fusion_method', type=str, default='concat')
    parser.add_argument('--sample_mode', type=str, default='Adaptive',choices=['random','GN','Adaptive','Bernoulli'])
    
    config = parser.parse_args()
    modal_rgb = [0]*config.epoch
    modal_of = [0]*config.epoch
    modal_depth = [0]*config.epoch

    if config.single_pretrain: is_pretrain = 'Pretrain'
    else: is_pretrain = 'NoPretrain'

    if config.mask_ffn and config.mask_resnet: use_ffn = 'AMSS'
    elif config.mask_ffn and not config.mask_resnet: use_ffn = 'NoBone'
    elif not config.mask_ffn and config.mask_resnet: use_ffn = 'NoFFN'
    else: use_ffn = 'Normal'

    if config.isbias==1: isbias='Enhance'
    else: isbias ='Origin'
    
    if use_ffn == 'Normal': config.our_model = 'lr{}_'.format(config.lr)
    else: config.our_model = 'lr{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(config.lr,isbias,config.temperature,config.optimizer,config.fusion_method,\
                                                           is_pretrain,use_ffn,config.sample_mode,config.patience,config.our_model)

    target_directory =  config.dataset+'/'+config.our_model
    if os.path.exists(target_directory):
    # 检查目标是否为文件夹
        if os.path.isdir(target_directory):
            # 删除目标文件夹及其内容
            shutil.rmtree(target_directory)
            print("文件夹已成功删除。")
        else:
            print("目标不是文件夹。")
    else:
        print("目标目录不存在。")
    writer = SummaryWriter(target_directory)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device_ids = [0,1]

    run(config)
    # test(config)

