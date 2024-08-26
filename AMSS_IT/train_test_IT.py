# -*- coding: utf-8 -*-
import argparse
import csv
import math
import shutil
import sys
import json
import time
import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from datetime import datetime
from fvcore.nn import FlopCountAnalysis, parameter_count
from ITModel import JointClsModel
from MyHelper import get_IT_data_loaders
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,average_precision_score
from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP

import random


sys.path.append('..')
sys.path.append('../..')
def create_folder_if_not_exists(path):
    """
    Create a folder if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def modal_caculate(a_out,v_out,label):

    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()
    score_v = sum([softmax(v_out)[i][torch.argmax(label[i]).item()] for i in range(v_out.size(0))])
    score_a = sum([softmax(a_out)[i][torch.argmax(label[i]).item()] for i in range(a_out.size(0))])

    ratio_v = score_v / score_a   # 反应visual模态与audio模态的强弱比
    ratio_a = 1 / ratio_v

    if ratio_v > 1:
        coeff_i = 1 - tanh(config.alpha * relu(ratio_v))
        coeff_t = 1
    else:
        coeff_t = 1 - tanh(config.alpha * relu(ratio_a))
        coeff_i = 1

    return coeff_t,coeff_i

def entropy(probabilities,epsilon=1e-8):
    # 将概率转换为对数概率
    log_probabilities = torch.log2(probabilities+ epsilon)
    
    # 计算信息熵
    entropy = -torch.sum(probabilities * log_probabilities)
    
    return entropy

def modal_caculate_MI(a_out,v_out,a,v,labels):
    def limit_number(num):
        if torch.isnan(num):
            return torch.tensor(0.5)
        return torch.clamp(num,0.1,0.9)

    label_single = torch.argmax(labels,1)
    label_counts = torch.bincount(label_single)
    label_probs = label_counts.float() / len(label_single)
    # 计算标签熵
    label_probs = label_probs.unsqueeze(0)
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    HY = entropy(label_probs)
    T = config.temperature
    score_v = -criterion(v_out,label_single)
    score_a = -criterion(a_out,label_single)
    pa = softmax(a)
    pv = softmax(v)
    Ha = entropy(pa)
    Hv = entropy(pv)

    if (score_a + HY + 0.5) < 0 or (score_v + HY + 0.5) <= 0:
        coeff_t = torch.tensor(0.5)
        coeff_i = torch.tensor(0.5)
        ratio_a = torch.tensor(1)
        ratio_v = torch.tensor(1)
        t_v = 1
        t_a = 1
    else:
        ratio_v = (score_v + HY) 
        ratio_a = (score_a + HY) 
        t_v = ratio_v / T
        t_a = ratio_a / T
        coeff_t = torch.exp(t_v) / (torch.exp(t_v) + torch.exp(t_a))
        coeff_i = torch.exp(t_a) / (torch.exp(t_v) + torch.exp(t_a))
        coeff_t = limit_number(coeff_t)
        coeff_i = limit_number(coeff_i)
    
    return coeff_t, coeff_i,ratio_a, ratio_v, t_a, t_v, HY

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)    
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_outputs(t, i, fusion_method, model):
        # Calculate outputs based on the fusion method
        if fusion_method == 'sum':
            return (
                torch.mm(i, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) + model.fusion_module.fc_y.bias,
                torch.mm(t, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) + model.fusion_module.fc_x.bias
            )
        elif fusion_method == 'concat':
            return (
                torch.mm(i, torch.transpose(model.fusion_module.fc_out.weight[:, 256:], 0, 1)) + model.fusion_module.fc_out.bias / 2,
                torch.mm(t, torch.transpose(model.fusion_module.fc_out.weight[:, :256], 0, 1)) + model.fusion_module.fc_out.bias / 2
            )
        else:
            a_zero_vector = torch.zeros_like(t)
            v_zero_vector = torch.zeros_like(i)
            return model.fusion_module(a_zero_vector, i), model.fusion_module(t, v_zero_vector)

def train_epoch(config, model, criterion, optimizer, train_loader,scheduler,epoch):
    
    model.train()
    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []
    train_tqdm = tqdm(train_loader)
    weight_size = 256 * 2
    num_av = [0.0 for _ in range(config.n_classes)]
    acc_a = [0.0 for _ in range(config.n_classes)]
    acc_v = [0.0 for _ in range(config.n_classes)]
    imbalance_ratio_path = 'mask_ratio/{}.tsv'.format(config.our_model)
    
    tsv_f = open(imbalance_ratio_path, 'a+', newline='', encoding='utf-8')
    tsv_w = csv.writer(tsv_f, delimiter='\t')
    
    for index,batch in enumerate(train_tqdm):
        optimizer.zero_grad()
        vdata_bert, vdata_img, label = batch
        t,i,out = model(vdata_bert.cuda(), vdata_img.cuda())
        t_out, i_out = calculate_outputs(t, i, config.fusion_method, model)
        if config.param_mode == 'fix':
            if config.fix_ratio == 1:   
                coeff_t = config.coeff_t
                coeff_i = config.coeff_i
            else:
                coeff_t, coeff_i = 0.5, 0.5
        else:
            coeff_t, coeff_i, score_t, score_i, ratio_t, ratio_i, HY = modal_caculate_MI(t_out, i_out, t, i, label.cuda())
        
        coeff_t = coeff_t.item() if torch.is_tensor(coeff_t) else coeff_t
        coeff_i = coeff_i.item() if torch.is_tensor(coeff_i) else coeff_i

        tsv_w.writerow([coeff_t, coeff_i])

        loss = criterion(out, label.cuda())
        label_single = torch.argmax(label,1)

        if coeff_t>coeff_i:
            modal_a[epoch]+=1
        else:
            modal_v[epoch]+=1
        
        if config.tensorboard == 1:
            writer.add_scalars('coeff', {'text': coeff_t}, 1+index + epoch*len(train_loader))
            writer.add_scalars('coeff',{'img': coeff_i}, 1+index + epoch*len(train_loader))

        loss.backward()
        multihead = 12
        
        for name, parms in model.named_parameters():
            if config.mask_ffn and('fc_out.weight' in name):
                gn = parms.grad.data ** 2
                wn = parms ** 2
                if config.gn_mode =='gn_wn':
                    mask_miu = (gn/wn).cuda()
                elif config.gn_mode =='gn':
                    mask_miu = (gn).cuda()
                elif config.gn_mode =='sqrt_gn':
                    mask_miu = torch.sqrt(gn).cuda()
                gn_audio = (mask_miu)[:, :weight_size // 2] #audio
                gn_video = (mask_miu)[:, weight_size // 2:]  #video
                p_a = math.ceil(gn_audio.size(1)*coeff_t)
                p_v = math.ceil(gn_video.size(1)*coeff_i)
                # print(p_a,p_v)
                if config.sample_mode == 'GN':
                    audio_fcn_indices = torch.argsort(gn_audio)[:,-p_a:]
                    video_fcn_indices = torch.argsort(gn_video)[:,-p_v:]
                    # print('GN')
                elif config.sample_mode == 'random':
                    audio_fcn_indices = np.zeros((gn_audio.size(0), p_a),dtype=int)
                    for a_i in range(gn_audio.size(0)):
                        audio_fcn_indices[a_i] = np.random.choice(range(gn_audio.size(1)), size=p_a, replace=False)
                    audio_fcn_indices = torch.from_numpy(audio_fcn_indices) #Mask random
                    # print('random')

                    video_fcn_indices = np.zeros((gn_video.size(0), p_v),dtype=int)
                    for v_i in range(gn_video.size(0)):
                        video_fcn_indices[v_i] = np.random.choice(range(gn_video.size(1)), size=p_v, replace=False)
                    video_fcn_indices = torch.from_numpy(video_fcn_indices) #Mask random
                
                elif config.sample_mode == 'Adaptive':
                    audio_importance = gn_audio / gn_audio.sum()
                    video_importance = gn_video / gn_video.sum()
                    audio_fcn_indices = torch.zeros((gn_audio.size(0), p_a),dtype=int)
                    for a_i in range(gn_audio.size(0)):
                        audio_fcn_indices[a_i] = torch.multinomial(audio_importance[a_i],num_samples=p_a, replacement=False)
                    video_fcn_indices = torch.zeros((gn_video.size(0), p_v),dtype=int)
                    for v_i in range(gn_video.size(0)):
                        video_fcn_indices[v_i] = torch.multinomial(video_importance[v_i],num_samples=p_v, replacement=False)
                
                mask_a_fcn = torch.zeros_like(gn_audio)
                mask_v_fcn = torch.zeros_like(gn_video)
                
                    
                for a_i,slice in enumerate(audio_fcn_indices):
                    if config.isbias == 1:
                        # mask_a_fcn[a_i][slice] = 1/(audio_importance[a_i][slice] + config.bias) # 将最大的 p 个数的位置赋值为 1
                        mask_a_fcn[a_i][slice] = 1/(audio_importance[a_i][slice] + 1/len(audio_importance)+ config.bias) # 将最大的 p 个数的位置赋值为 1
                    elif config.isbias == 2:
                        mask_a_fcn[a_i][slice] = 1/(audio_importance[a_i][slice] + config.bias) # 将最大的 p 个数的位置赋值为 1
                    else:
                        mask_a_fcn[a_i][slice]=1 
                for v_i,slice in enumerate(video_fcn_indices):
                    if config.isbias == 1:
                        # mask_v_fcn[v_i][slice] = 1/(video_importance[v_i][slice] + config.bias) # 将最大的 p 个数的位置赋值为 1
                        mask_v_fcn[v_i][slice] = 1/(video_importance[v_i][slice] + 1/len(video_importance)+ config.bias) # 将最大的 p 个数的位置赋值为 1
                    elif config.isbias == 2:
                        mask_v_fcn[v_i][slice] = 1/(video_importance[v_i][slice] + config.bias) # 将最大的 p 个数的位置赋值为 1
                    else:
                        mask_v_fcn[v_i][slice]=1 
                parms.grad[:, :weight_size // 2] *= mask_a_fcn
                parms.grad[:, weight_size // 2:] *= mask_v_fcn

            if config.mask_resnet and 'text'in name and 'attention' in name and 'layer' in name and 'weight' in name and  len(parms.grad.size()) == 2 :
                # print(name)
                gn = parms.grad.data ** 2#(grad ** 2).sum().item()
                wn = parms ** 2
                if config.gn_mode =='gn_wn':
                    mask_miu = (gn/wn).cuda()
                elif config.gn_mode =='gn':
                    mask_miu = (gn).cuda()
                elif config.gn_mode=='sqrt_gn':
                    mask_miu = torch.sqrt(gn).cuda()
                if 'output' in  name:
                    sub_matrices = torch.split(mask_miu, int((mask_miu.size(1))/multihead), dim=1)
                else:
                    sub_matrices = torch.split(mask_miu, int((mask_miu.size(0))/multihead), dim=0)
                sums = [torch.sum(sub_matrix, dim=(0,1)).item() for sub_matrix in sub_matrices]
                sums = torch.tensor(sums)
                # print(coeff_t)
                p_a = int(torch.ceil(len(sums)*torch.tensor(coeff_t)).item())
                if config.sample_mode == 'GN':
                    max_p_indices = torch.argsort(sums)[-p_a:]
                    # print(max_p_indices)
                elif config.sample_mode =='random':
                    max_p_indices = np.random.choice(range(len(sums)), p_a, replace=False)
                    max_p_indices = torch.from_numpy(max_p_indices) #Mask random
                    # print(max_p_indices)
                elif config.sample_mode == 'Bernoulli':
                    important_gn = sums/sum(sums)
                    max_p_indice = torch.bernoulli(important_gn)
                    max_p_indices  = torch.nonzero(max_p_indice).squeeze()
                elif config.sample_mode == 'Adaptive':
                    # important_gn = softmax_dim0(mask_miu)
                    important_gn = sums/sum(sums)
                    max_p_indices = torch.multinomial(important_gn,num_samples=p_a, replacement=False)
                    
                mask_a_indice = torch.zeros_like(sums)  # 创建与 n 相同大小的全零数组
                if config.isbias == 1:
                    # mask_a_indice[max_p_indices] = 1/(important_gn[max_p_indices] + config.bias) # 将最大的 p 个数的位置赋值为 1
                    mask_a_indice[max_p_indices] = 1/(important_gn[max_p_indices] + 1/len(important_gn)+ config.bias) # 将最大的 p 个数的位置赋值为 1
                elif config.isbias == 2:   
                    mask_a_indice[max_p_indices] = 1/(important_gn[max_p_indices] + config.bias) # 将最大的 p 个数的位置赋值为 1
                else:
                    mask_a_indice[max_p_indices] = 1  # 将最大的 p 个数的位置赋值为 1
                if 'output' in  name:
                    sub_matrices = [torch.full((int((mask_miu.size(0))),int((mask_miu.size(1))/multihead)), value) for value in mask_a_indice]
                    combined_matrix = torch.cat(sub_matrices, dim=1).cuda()

                else:
                    sub_matrices = [torch.full((int((mask_miu.size(0))/multihead),int(mask_miu.size(1))), value) for value in mask_a_indice]
                    combined_matrix = torch.cat(sub_matrices, dim=0).cuda()
                parms.grad *= combined_matrix

            if config.mask_resnet and 'img'in name and 'layer' in name and  'conv' in name and  len(parms.grad.size()) == 4 :
                gn = torch.sum(parms.grad.data ** 2, dim=(1, 2, 3))#(grad ** 2).sum().item()
                wn = torch.sum(parms ** 2, dim=(1, 2, 3))
                # print(name)
                if config.gn_mode =='gn_wn':
                    mask_miu = (gn/wn).cuda()
                elif config.gn_mode =='gn':
                    mask_miu = (gn).cuda()
                elif config.gn_mode=='sqrt_gn':
                    mask_miu = torch.sqrt(gn).cuda()
                p_v = math.ceil(len(mask_miu)*coeff_i)
                if config.sample_mode == 'GN':
                    max_p_indices = torch.argsort(mask_miu)[-p_v:]
                elif config.sample_mode =='random':
                    max_p_indices = np.random.choice(range(len(mask_miu)), p_v, replace=False)
                    max_p_indices = torch.from_numpy(max_p_indices) #Mask random
                elif config.sample_mode == 'Bernoulli':
                    important_gn = mask_miu/sum(mask_miu)
                    max_p_indice = torch.bernoulli(important_gn)
                    max_p_indices  = torch.nonzero(max_p_indice).squeeze()
                elif config.sample_mode == 'Adaptive':
                    # important_gn = softmax_dim0(mask_miu)
                    important_gn = mask_miu/mask_miu.sum()
                    max_p_indices = torch.multinomial(important_gn,num_samples=p_v, replacement=False)
                # max_p_indices = torch.argsort(mask_miu)[:p_v]
                mask_v = torch.zeros_like(mask_miu)  # 创建与 n 相同大小的全零数组
                if config.isbias == 1:
                    # mask_v[max_p_indices] = 1/(important_gn[max_p_indices] + config.bias)  # 将最大的 p 个数的位置赋值为 1
                    mask_v[max_p_indices] = 1/(important_gn[max_p_indices] + 1/len(important_gn)+ config.bias)  # 将最大的 p 个数的位置赋值为 1
                elif config.isbias == 2:
                    mask_v[max_p_indices] = 1/(important_gn[max_p_indices] + config.bias) 
                else:    
                    mask_v[max_p_indices] = 1  # 将最大的 p 个数的位置赋值为 1
                # mask_v[max_p_indices] = 1  # 将最大的 p 个数的位置赋值为 1
                parms.grad *= mask_v.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(parms.grad)
        
        optimizer.step()
        if config.tensorboard == 1:
            writer.add_scalars('coeff', {'audio': coeff_t}, 1+index + epoch*len(train_loader))
            writer.add_scalars('coeff',{'video': coeff_i}, 1+index + epoch*len(train_loader))
        temp_out = out.detach().cpu()
        preds = torch.argmax(temp_out, dim=1)
        all_score += preds.numpy().tolist()
        all_out += out.detach().cpu().numpy().tolist()
        all_label += label.cpu().numpy().tolist()
        all_loss.append(loss.item())
        all_label_single += label_single.detach().cpu().tolist()

        # Update tqdm description
        train_tqdm.set_description(
            'Loss: %.3f coeff_t: %.3f coeff_i: %.3f Ht: %.3f Hi: %.3f' % (
                np.mean(all_loss), coeff_t, coeff_i, score_t, score_i
            )
        )
    tsv_f.close()
    # Update learning rate scheduler if applicable
    if config.fusion_method != 'film' and config.optimizer == 'SGD':
        scheduler.step()

    # Log training accuracy to TensorBoard if enabled
    if config.tensorboard == 1:
        writer.add_scalars('train_acc_av', {'acc_a': sum(acc_a) / sum(num_av) * 100}, epoch + 1)
        writer.add_scalars('train_acc_av', {'acc_v': sum(acc_v) / sum(num_av) * 100}, epoch + 1)

    # Calculate and return average loss and accuracy
    acc = accuracy_score(all_label_single, all_score) * 100
    return np.mean(all_loss), acc

def evaluate(config, model, criterion, valid_loader,epoch):

    model.eval()
    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []
    softmax = nn.Softmax(dim=1)

    # Initialize counters for class-wise accuracy
    num = [0.0 for _ in range(config.n_classes)]
    acc_t = [0.0 for _ in range(config.n_classes)]
    acc_i = [0.0 for _ in range(config.n_classes)]

    # Get the length of the validation loader
    valid_len = len(valid_loader)

    for index,batch in enumerate(tqdm(valid_loader)):
        with torch.no_grad():
            # Unpack the batch
            vdata_bert, vdata_img, label = batch
            
            # Forward pass through the model
            t, i, out = model(vdata_bert.cuda(), vdata_img.cuda())
            
            # Calculate the outputs based on fusion method
            t_out, i_out = calculate_outputs(t, i, config.fusion_method, model)
            
            # Get the single-label version of the labels
            label_single = torch.argmax(label, dim=1)

            # Compute loss
            loss = criterion(out, label_single.cuda())
            
            # Get predictions
            pred_v = softmax(i_out)
            pred_a = softmax(t_out)

            # Update counters and accuracy metrics
            for i in range(vdata_img.shape[0]):
                class_label = label_single[i].cpu().item()
                num[class_label] += 1.0
                
                v_pred = np.argmax(pred_v[i].cpu().data.numpy())
                a_pred = np.argmax(pred_a[i].cpu().data.numpy())
                
                if class_label == v_pred:
                    acc_i[class_label] += 1.0
                if class_label == a_pred:
                    acc_t[class_label] += 1.0
             # Collect outputs for metrics calculation
            all_loss.append(loss.item())
            tmp_out = out.detach().cpu()
            preds = torch.argmax(tmp_out, dim=1)
            prediction = softmax(tmp_out)
        all_out += prediction.numpy().tolist()
        all_score += preds.numpy().tolist()
        all_label += label.cpu().numpy().tolist()
        all_label_single += label_single.cpu().tolist()
        
        # Log test loss
        if config.tensorboard == 1:
            writer.add_scalars('test_loss', {'test_loss': loss.item()}, index + valid_len * epoch)
    
    if config.tensorboard == 1:
        writer.add_scalars('test_acc_av', {'acc_t': sum(acc_t) / sum(num) * 100}, epoch + 1)
        writer.add_scalars('test_acc_av', {'acc_i': sum(acc_i) / sum(num) * 100}, epoch + 1)
    # Calculate metrics
    text_acc = sum(acc_t) / sum(num) * 100
    img_acc = sum(acc_i) / sum(num) * 100
    auc = roc_auc_score(all_label, all_out, multi_class='ovo') * 100
    acc = accuracy_score(all_label_single, all_score) * 100

    # Average Precision Score
    AP = []
    all_label = np.array(all_label)
    all_out = np.array(all_out)
    for i in range(len(all_label[0])):
        AP.append(average_precision_score(all_label[:, i], all_out[:, i]))

    # F1 Score
    f1 = f1_score(all_label_single, all_score, average='macro') * 100

    # Return metrics
    return {
        'loss': np.mean(all_loss),
        'acc': acc,
        't_acc': text_acc,
        'i_acc': img_acc,
        'auc': auc,
        'f1': f1,
        'mAP': np.mean(AP) * 100
    }

def train_test(config, model, model_path, train_loader, valid_loader):

    if config.dataset in ['Sarcasm', 'CrisisMMD']:
        if config.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
        elif config.optimizer == 'AdaGrad':
            optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
        elif config.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99))
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.patience, 0.1)

    elif config.dataset == 'Twitter15':
        total_steps = len(train_loader) * config.epoch
        
        if config.optimizer == 'SGD':
            print('Use SGD')
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        else:
            optimizer = AdamW(model.parameters(), lr=config.lr, correct_bias=True)
            if config.fusion_method == 'film':
                scheduler = optim.lr_scheduler.StepLR(optimizer, config.patience, 0.1)
            else:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    current_date = datetime.now()
    
    # 创建文件夹名为月份加日期，例如：11.22
    folder_name = current_date.strftime('%m.%d')
    # Prepare to log results
    results_path = f'results/{config.dataset}/{folder_name}'
    create_folder_if_not_exists(results_path)
    results_path += f'/{config.our_model}_acc_seed{config.random_seed}.tsv'
    with open(results_path, 'a+', newline='', encoding='utf-8') as tsv_f:
        tsv_w = csv.writer(tsv_f, delimiter='\t')
        tsv_w.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'text_acc', 'img_acc', 'ACC', 'AUC', 'F1', 'mAP', 'Train_Time'])

    for epoch in range(config.epoch):
        model.zero_grad()
        print(f'Training Epoch: {epoch}')
        start_time = time.time()

        train_loss, train_acc = train_epoch(config, model, criterion, optimizer, train_loader, scheduler, epoch)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time} seconds")

        valid_result = evaluate(config, model, criterion, valid_loader, epoch)

        print(f'epoch: {epoch} train_loss: {train_loss} Train_ACC: {train_acc} ACC: {valid_result["acc"]} AUC: {valid_result["auc"]} Mac-F1: {valid_result["f1"]} mAP: {valid_result["mAP"]}')

        with open(results_path, 'a+', newline='', encoding='utf-8') as tsv_f:
            tsv_w = csv.writer(tsv_f, delimiter='\t')
            tsv_w.writerow([epoch, train_loss, train_acc, valid_result['loss'], valid_result['t_acc'], valid_result['i_acc'], valid_result['acc'], valid_result['auc'], valid_result['f1'], valid_result['mAP'], training_time])

        if valid_result['acc'] > best_acc:
            best_acc = valid_result['acc']
            torch.save(model, model_path)
    # model = torch.load(model_path)

def run(config):

    # Set random seed for reproducibility
    set_seed(config.random_seed)

    # Determine the number of available GPUs
    config.n_gpu = torch.cuda.device_count()
    print("Num GPUs:", config.n_gpu)
    print("Device:", config.device)

    # Load training and testing data
    train_loader, test_loader = get_IT_data_loaders(config)

    # Define model names and their corresponding paths
    models_name = [config.img_pretrain, config.text_pretrain, config.our_model]
    models_path = [os.path.join('checkpoint', config.dataset, name + '.pt') for name in models_name]

    print('-' * 20)

    # Set the number of classes based on the dataset
    if config.dataset in ['Sarcasm', 'CrisisMMD']:
        config.n_classes = 2
    elif config.dataset == 'Twitter15':
        config.n_classes = 3

    print(config.dataset)

    print('Now is modal of img&text')
    model_path = models_path[2]
    model = JointClsModel(config)
    model.cuda()

    # if config.cal_complex:
    #     print('---caculate model complex---')
    #     vdata_bert = torch.randint(0, 10000, (32, 128))  # batch_size=32, sequence_length=128
    #     vdata_img = torch.randn(32, 3, 224, 224) 
    #     flops = FlopCountAnalysis(model, (vdata_bert.cuda(), vdata_img.cuda()))
    #     params = parameter_count(model)
    #     print("FLOPs %.2f M:" % (flops.total()/ 1000000.0))
    #     print("Params %.2f M: " %(params['']/ 1000000.0))
    #     exit()
    # Determine if a pre-trained model should be used
    if config.single_pretrain == 1:
        print('Use Pre-Train Model')
    else:
        print('Don\'t Use Pre-Train Model')

    # Ensure the checkpoint folder exists
    folder_path = os.path.join('checkpoint', config.dataset)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'{folder_path} created successfully.')
    else:
        print(f'{folder_path} already exists.')

    # Move the model to the GPU if available

    # Evaluate the model if only testing is enabled
    # Train and test the model
    train_test(config, model, model_path, train_loader, test_loader)
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Text-Img Dataset Config')

    # basic settings
    # Argument parser setup
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU for training.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--data_path', type=str, default='/media/php/data/data_processing', help='Path to the data directory.')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length.')
    parser.add_argument('--pre_train', type=bool, default=False, help='Whether to use pre-trained models.')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='Name of the BERT model.')
    parser.add_argument('--modal', type=str, default='multi', choices=['multi','film', 'gated', 'concat', 'ORG_GE', 'img', 'text', 'GB', 'cmml'], help='Modal fusion method.')
    parser.add_argument('--only_test', type=int, default=0, help='Whether to only test the model.')
    parser.add_argument('--warm_up_epoch', type=int, default=0, help='Number of warm-up epochs.')
    parser.add_argument('--epoch', type=int, default=70, help='Total number of training epochs.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers.')
    parser.add_argument('--img_lr', type=float, default=0.00001, help='Learning rate for image data.')
    parser.add_argument('--text_lr', type=float, default=0.00001, help='Learning rate for text data.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for multi-modal data.')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature parameter for scaling.')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use.')
    parser.add_argument('--ratio_random', type=int, default=0, help='Random ratio for sampling.')
    parser.add_argument('--bias', type=float, default=0, help='Bias value for mask calculations.')
    parser.add_argument('--isbias', type=int, default=0, help='Flag to indicate bias usage.')

    parser.add_argument('--sample_mode', type=str, default='random', choices=['random', 'GN', 'Adaptive', 'Bernoulli'], help='Sampling mode for feature masking.')
    parser.add_argument('--gn_mode', type=str, default='gn', choices=['gn', 'gn_wn', 'sqrt_gn'], help='Gradient normalization mode.')
    parser.add_argument('--mask_resnet', type=int, default=1, help='Whether to use mask for ResNet.')
    parser.add_argument('--mask_ffn', type=int, default=1, help='Whether to use mask for FFN.')
    parser.add_argument('--alpha', type=float, default=0.4, help='Alpha parameter for modulation.')

    parser.add_argument('--gimma', type=int, default=1, help='Flag for using GIMMA.')
    parser.add_argument('--single_pretrain', type=int, default=0, help='Flag for single pre-training.')

    parser.add_argument('--weight_decay_text', type=float, default=0.0001, help='Weight decay for text parameters.')
    parser.add_argument('--weight_decay_img', type=float, default=0.0001, help='Weight decay for image parameters.')
    parser.add_argument('--weight_decay_multi', type=float, default=0.0001, help='Weight decay for multi-modal parameters.')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for all parameters.')
    parser.add_argument('--modulation_ends', type=int, default=30, help='Epoch at which modulation ends.')
    parser.add_argument('--coeff', type=str, default='parameter_sensitive', help='Coefficient type for parameter sensitivity.')

    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--dataset', type=str, default='Sarcasm', help='Name of the dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--single_epoch', type=int, default=15, help='Number of epochs for single model training.')
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size for testing.')

    parser.add_argument('--n_classes', type=int, default=31, help='Number of output classes.')
    parser.add_argument('--text_dim', type=int, default=300, help='Dimensionality of text features.')
    parser.add_argument('--feature_dim', type=int, default=512, help='Dimensionality of features.')

    parser.add_argument('--fusion_method', type=str, default='concat', choices=['concat', 'mean', 'max'], help='Method for feature fusion.')
    parser.add_argument('--single_pre_train', action='store_true', help='Flag for single pre-training.')
    parser.add_argument('--img_pretrain', type=str, default="", help='Path to image pre-trained model.')
    parser.add_argument('--text_pretrain', type=str, default="", help='Path to text pre-trained model.')
    parser.add_argument('--our_model', type=str, default="SEBM", help='Name of the custom model.')
    parser.add_argument('--test_only', dest='test_only', action='store_true', help='Flag to only test the model.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cuda or cpu).')
    parser.add_argument('--gpu_id', type=str, default='0', help='ID(s) for CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--tensorboard', type=int, default=0, help='Whether to save logs to TensorBoard.')
    parser.add_argument('--param_mode', type=str, default='dynamic', help='Parameter mode for dynamic adjustments.')
    parser.add_argument('--cal_complex', type=int, default=1, help='Parameter mode for dynamic adjustments.')


    config = parser.parse_args()

    modal_v = [0] * config.epoch
    modal_a = [0] * config.epoch

    if config.mask_ffn and config.mask_resnet:
        use_ffn = 'AMSS'
    elif config.mask_ffn and not config.mask_resnet:
        use_ffn = 'NoBone'
    elif not config.mask_ffn and config.mask_resnet:
        use_ffn = 'NoFFN'
    else:
        use_ffn = 'Normal'
        
    isbias = 'AMSS+' if config.isbias == 1 else '(a)' if config.isbias == 2 else 'AMSS'
    if not config.mask_ffn and not config.mask_resnet:
        isbias = 'Normal'
        
    if config.param_mode == 'fix':
        if config.fix_ratio == 1:
            config.our_model = f"A{config.coeff_t}_V{config.coeff_i}_{config.lr}"
        else:
            coeff_t, coeff_i = 0.5, 0.5
    else:
        config.our_model = f"{isbias}_lr{config.lr}_T{config.temperature}_{config.optimizer}_{config.fusion_method}_{use_ffn}_{config.sample_mode}_{config.gn_mode}_{config.patience}_{config.our_model}"
    if config.tensorboard == 1:
        target_directory = f"{config.dataset}/{config.our_model}"
        if os.path.exists(target_directory) and os.path.isdir(target_directory):
            shutil.rmtree(target_directory)
            print("文件夹已成功删除。")
        elif not os.path.exists(target_directory):
            print("目标目录不存在。")
        else:
            print("目标不是文件夹。")
        writer = SummaryWriter(target_directory)


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id     
    run(config)
    
    # test(config)

