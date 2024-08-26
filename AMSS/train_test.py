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
from MaskModel import AVClassifier
from MMTM  import MMTMClassifier
from MyHelper import get_VA_data_loaders
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,average_precision_score
import os
import random
# import matplotlib.pyplot as plt
# from thop import profile
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
sys.path.append('..')
sys.path.append('../..')
import scipy.io.wavfile as wav
from scipy.stats import entropy
import cv2
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count
'''
强弱模态量化代码
'''

def modal_caculate_MI(a_out, v_out, a, v, labels):
    """
    Calculate mutual information coefficients for audio and visual modalities.

    Args:
        a_out (Tensor): Output of the audio model.
        v_out (Tensor): Output of the visual model.
        a (Tensor): Audio features.
        v (Tensor): Visual features.
        labels (Tensor): Ground truth labels.
        config (object): Configuration object containing temperature parameter.

    Returns:
        tuple: Coefficients and ratios for audio and visual modalities, temperature values, and entropy.
    """
    def limit_number(num):
        if torch.isnan(num):
            return torch.tensor(0.5)
        return torch.clamp(num, 0.1, 0.9)
    
    # Determine the ground truth labels
    label_single = torch.argmax(labels, dim=1)
    
    # Compute label probabilities
    label_counts = torch.bincount(label_single)
    label_probs = label_counts.float() / len(label_single)
    
    # Initialize loss and activation functions
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    
    # Compute entropy of the labels
    HY = criterion(label_probs, label_probs)
    
    # Compute scores for audio and visual outputs
    score_v = -criterion(v_out, label_single)
    score_a = -criterion(a_out, label_single)
    
    # Compute probabilities from logits
    pa = softmax(a)
    pv = softmax(v)
    
    # Compute entropy of audio and visual features
    Ha = criterion(a, pa)
    Hv = criterion(v, pv)
    
    # Calculate ratios and coefficients
    T = config.temperature
    if (score_a + HY + 0.5) <= 0 or (score_v + HY + 0.5) <= 0:
        coeff_a, coeff_v, ratio_a, ratio_v, t_v, t_a = torch.tensor(0.5), torch.tensor(0.5), torch.tensor(1.0), torch.tensor(1.0), 1.0, 1.0
    else:
        ratio_v = (score_v + HY + 0.5) / Hv
        ratio_a = (score_a + HY + 0.5) / Ha
        t_v = ratio_v / T
        t_a = ratio_a / T
        coeff_a = torch.exp(t_v) / (torch.exp(t_v) + torch.exp(t_a))
        coeff_v = torch.exp(t_a) / (torch.exp(t_v) + torch.exp(t_a))
        coeff_a, coeff_v = limit_number(coeff_a), limit_number(coeff_v)
    
    return coeff_a, coeff_v, ratio_v, ratio_a, t_v, t_a, HY

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)    
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_outputs(a, v, fusion_method, model):
        # Calculate outputs based on the fusion method
        if fusion_method == 'sum':
            return (
                torch.mm(v, torch.transpose(model.fusion_module.fc_y.weight, 0, 1)) + model.fusion_module.fc_y.bias,
                torch.mm(a, torch.transpose(model.fusion_module.fc_x.weight, 0, 1)) + model.fusion_module.fc_x.bias
            )
        elif fusion_method == 'mmtm':
            return (
                torch.mm(v, torch.transpose(model.video_fc.weight, 0, 1)) + model.video_fc.bias,
                torch.mm(a, torch.transpose(model.auido_fc.weight, 0, 1)) + model.auido_fc.bias
            )
        elif fusion_method == 'concat':
            return (
                torch.mm(v, torch.transpose(model.fusion_module.fc_out.weight[:, 512:], 0, 1)) + model.fusion_module.fc_out.bias / 2,
                torch.mm(a, torch.transpose(model.fusion_module.fc_out.weight[:, :512], 0, 1)) + model.fusion_module.fc_out.bias / 2
            )
        else:
            a_zero_vector = torch.zeros_like(a)
            v_zero_vector = torch.zeros_like(v)
            return model.fusion_module(a_zero_vector, v), model.fusion_module(a, v_zero_vector)
        
def train_epoch(config, model, criterion, optimizer, train_loader,scheduler,epoch):

    # Set model to training mode
    model.train()
    
    # Initialize lists to store various metrics
    all_label, all_out, all_loss, all_score, all_label_single = [], [], [], [], []

    num_av, acc_a, acc_v = [0.0] * config.n_classes, [0.0] * config.n_classes, [0.0] * config.n_classes
    
    # Initialize progress bar for training
    train_tqdm = tqdm(train_loader)
    
    # Define softmax for later use
    softmax = nn.Softmax(dim=1)

    
    for index,batch in enumerate(train_tqdm):
        optimizer.zero_grad()
        spectrogram, images, label = batch
        # print(spectrogram.unsqueeze(1).float().shape, images.float().shape)
        # Forward pass through the model
        a, v, out = model(spectrogram.unsqueeze(1).float().cuda(), images.float().cuda())
        
        # Calculate outputs based on fusion method
        v_out, a_out = calculate_outputs(a, v, config.fusion_method, model)
        
        # Calculate mutual information and other metrics
        if config.param_mode == 'fix':
            if config.fix_ratio == 1:   
                coeff_v = config.coeff_v
                coeff_a = config.coeff_a
            else:
                coeff_a, coeff_v = 0.5, 0.5
        else:
            coeff_a, coeff_v, score_v, score_a, ratio_v, ratio_a, HY = modal_caculate_MI(a_out, v_out, a, v, label.cuda())
        
        coeff_a = coeff_a.item() if torch.is_tensor(coeff_a) else coeff_a
        coeff_v = coeff_v.item() if torch.is_tensor(coeff_v) else coeff_v

        if coeff_a>coeff_v:
            modal_a[epoch]+=1
        else:
            modal_v[epoch]+=1

        # Convert labels to single class
        label_single = torch.argmax(label, 1)
        
        # Get softmax predictions
        pred_v = softmax(v_out)
        pred_a = softmax(a_out)

        for i in range(images.shape[0]):
            num_av[label_single[i].cpu()] += 1.0
            v_i = np.argmax(pred_v[i].cpu().data.numpy())
            a_i = np.argmax(pred_a[i].cpu().data.numpy())      
            if np.asarray(label_single[i].cpu()) == v_i:
                acc_v[label_single[i].cpu()] += 1.0
            if np.asarray(label_single[i].cpu()) == a_i:
                acc_a[label_single[i].cpu()] += 1.0

        loss = criterion(out,label_single.cuda()) 
        
        loss.backward()
        for name, parms in model.named_parameters():
            if config.mask_ffn and ('fc_out.weight' in name):
                gn = parms.grad.data ** 2
                wn = parms ** 2
                # Calculate mask_miu based on gn_mode
                if config.gn_mode == 'gn_wn':
                    mask_miu = (gn / wn).cuda()
                elif config.gn_mode == 'gn':
                    mask_miu = gn.cuda()
                elif config.gn_mode == 'sqrt_gn':
                    mask_miu = torch.sqrt(gn).cuda()
                
                # Split mask_miu into audio and video parts
                gn_audio = mask_miu[:, :1024 // 2]
                gn_video = mask_miu[:, 1024 // 2:]
                
                # Calculate the number of features to keep
                p_a = int(gn_audio.size(1) * coeff_a)
                p_v = int(gn_video.size(1) * coeff_v)
                
                # Sample indices based on sample_mode
                if config.sample_mode in ['TopK', 'Bernoulli']:
                    audio_fcn_indices = torch.argsort(gn_audio)[:,-p_a:]
                    video_fcn_indices = torch.argsort(gn_video)[:,-p_v:]
                elif config.sample_mode == 'random':
                    audio_fcn_indices = np.zeros((gn_audio.size(0), p_a),dtype=int)
                    for a_i in range(gn_audio.size(0)):
                        audio_fcn_indices[a_i] = np.random.choice(range(gn_audio.size(1)), size=p_a, replace=False)
                    audio_fcn_indices = torch.from_numpy(audio_fcn_indices) #Mask random

                    video_fcn_indices = np.zeros((gn_video.size(0), p_v),dtype=int)
                    for v_i in range(gn_video.size(0)):
                        video_fcn_indices[v_i] = np.random.choice(range(gn_video.size(1)), size=p_v, replace=False)
                    video_fcn_indices = torch.from_numpy(video_fcn_indices) #Mask random
                
                elif config.sample_mode == 'Adaptive':
                    audio_importance = gn_audio / gn_audio.sum(dim=1, keepdim=True)
                    video_importance = gn_video / gn_video.sum(dim=1, keepdim=True)
                    audio_fcn_indices = torch.multinomial(audio_importance, num_samples=p_a, replacement=False)
                    video_fcn_indices = torch.multinomial(video_importance, num_samples=p_v, replacement=False)
                
                # Create masks for audio and video features
                mask_a_fcn = torch.zeros_like(gn_audio)
                mask_v_fcn = torch.zeros_like(gn_video)
                
                for i, slice in enumerate(audio_fcn_indices):
                    mask_a_fcn[i][slice] = 1 if config.isbias == 0 else 1 / (audio_fcn_indices[i] + 1 / len(audio_fcn_indices))
                for i, slice in enumerate(video_fcn_indices):
                    mask_v_fcn[i][slice] = 1 if config.isbias == 0 else 1 / (video_fcn_indices[i] + 1 / len(video_fcn_indices))
                
            if config.mask_resnet and 'audio'in name and 'layer' in name and 'conv' in name and  len(parms.grad.size()) == 4 :
                gn = torch.sum(parms.grad.data ** 2, dim=(1, 2, 3))
                wn = torch.sum(parms ** 2, dim=(1, 2, 3))
                
                if config.gn_mode == 'gn_wn':
                    mask_miu = (gn / wn).cuda()
                elif config.gn_mode == 'gn':
                    mask_miu = gn.cuda()
                elif config.gn_mode == 'sqrt_gn':
                    mask_miu = torch.sqrt(gn).cuda()
                
                p_a = math.ceil(len(mask_miu) * coeff_a)

                if config.sample_mode == 'TopK':
                    max_p_indices = torch.argsort(mask_miu, descending=True)[:p_a]
                elif config.sample_mode == 'random':
                    max_p_indices = torch.from_numpy(np.random.choice(range(len(mask_miu)), p_a, replace=False))
                elif config.sample_mode == 'Bernoulli':
                    important_gn = mask_miu / mask_miu.sum()
                    max_p_indices = torch.nonzero(torch.bernoulli(important_gn)).squeeze()
                elif config.sample_mode == 'Adaptive':
                    important_gn = mask_miu / mask_miu.sum()
                    max_p_indices = torch.multinomial(important_gn, num_samples=p_a, replacement=False)
                
                mask_a = torch.zeros_like(mask_miu)  # 创建与 n 相同大小的全零数组
                
                if config.isbias == 1 and config.sample_mode == 'Adaptive':
                    mask_a[max_p_indices] = 1/(important_gn[max_p_indices] + 1/len(important_gn)) # 将最大的 p 个数的位置赋值为 1
                else:    
                    mask_a[max_p_indices] = 1  # 将最大的 p 个数的位置赋值为 1
                
                parms.grad *= mask_a.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(parms.grad)                

            if config.mask_resnet and 'video'in name and 'layer' in name and  'conv' in name and  len(parms.grad.size()) == 4 :
                gn = torch.sum(parms.grad.data ** 2, dim=(1, 2, 3))
                wn = torch.sum(parms ** 2, dim=(1, 2, 3))
                
                if config.gn_mode == 'gn_wn':
                    mask_miu = (gn / wn).cuda()
                elif config.gn_mode == 'gn':
                    mask_miu = gn.cuda()
                elif config.gn_mode == 'sqrt_gn':
                    mask_miu = torch.sqrt(gn).cuda()
                
                p_v = math.ceil(len(mask_miu) * coeff_v)

                if config.sample_mode == 'TopK':
                    max_p_indices = torch.argsort(mask_miu, descending=True)[:p_v]
                elif config.sample_mode == 'random':
                    max_p_indices = torch.from_numpy(np.random.choice(range(len(mask_miu)), p_v, replace=False))
                elif config.sample_mode == 'Bernoulli':
                    important_gn = mask_miu / mask_miu.sum()
                    max_p_indices = torch.nonzero(torch.bernoulli(important_gn)).squeeze()
                elif config.sample_mode == 'Adaptive':
                    important_gn = mask_miu / mask_miu.sum()
                    max_p_indices = torch.multinomial(important_gn, num_samples=p_v, replacement=False)
                
                mask_v = torch.zeros_like(mask_miu)  # 创建与 n 相同大小的全零数组

                if config.isbias == 1 and config.sample_mode == 'Adaptive':
                    mask_v[max_p_indices] = 1/(important_gn[max_p_indices] + 1/len(important_gn)) # 将最大的 p 个数的位置赋值为 1

                else:    
                    mask_v[max_p_indices] = 1  # 将最大的 p 个数的位置赋值为 1

                parms.grad *= mask_v.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand_as(parms.grad)
       
        optimizer.step()
        if config.tensorboard == 1:
            metrics = {
                'coeff': {'audio': coeff_a, 'video': coeff_v},
                'ratio': {'audio': ratio_v, 'video': ratio_a},
                'score': {'audio': score_v, 'video': score_a}
            }
            for key, value in metrics.items():
                writer.add_scalars(key, value, 1 + index + epoch * len(train_loader))

        temp_out = out.detach().cpu()
        preds = torch.argmax(temp_out, 1)
        all_score += preds.numpy().tolist()
        all_out += temp_out.numpy().tolist()
        all_label += label
        all_label_single += label_single.detach().cpu().tolist()
        all_loss.append(loss.item())

        train_tqdm.set_description(
            'Loss: %.3f, coeff_a: %.3f, coeff_v: %.3f, ratio_v: %.3f, ratio_a: %.3f' % 
            (np.mean(all_loss), coeff_a, coeff_v, ratio_v, ratio_a)
        )
        
        
        if config.tensorboard == 1:       
            train_acc_av = {
            'acc_a': sum(acc_a) / sum(num_av) * 100,
            'acc_v': sum(acc_v) / sum(num_av) * 100
            }
            writer.add_scalars('train_acc_av', train_acc_av, epoch + 1)    
    scheduler.step()
    acc = accuracy_score(all_label_single, all_score)*100
    # print(a_features.shape,v_features.shape)
    return  np.mean(all_loss), acc
    

def evaluate(config, model, criterion, optimizer, valid_loader, epoch, warm_up=False, coeff_a=0.5, coeff_v=0.5):

    model.eval()
    # Initialize tracking variables
    all_out, all_score, all_label, all_loss = [], [], [], []
    cls_loss_a, cls_loss_v = [], []
    all_label_single = []
    softmax = nn.Softmax(dim=1)
    num = [0.0 for _ in range(config.n_classes)]
    acc_a = [0.0 for _ in range(config.n_classes)]
    acc_v = [0.0 for _ in range(config.n_classes)]
    # 获取验证集长度
    valid_len = len(valid_loader)
    valid_tqdm = tqdm(valid_loader)

    for index, batch in enumerate(valid_tqdm):
        with torch.no_grad():
            spectrogram, images, label = batch
            # print(spectrogram.shape)
            a, v, out = model(spectrogram.unsqueeze(1).float().cuda(), images.float().cuda())
        
            # Calculate outputs based on fusion method
            v_out, a_out = calculate_outputs(a, v, config.fusion_method, model)

            label_single = torch.argmax(label,1)
            pred_v = softmax(v_out)
            pred_a = softmax(a_out)
            for i in range(images.shape[0]):
                num[label_single[i].cpu()] += 1.0
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())      
                if np.asarray(label_single[i].cpu()) == v:
                    acc_v[label_single[i].cpu()] += 1.0
                if np.asarray(label_single[i].cpu()) == a:
                    acc_a[label_single[i].cpu()] += 1.0
            loss = criterion(out, label_single.cuda())    
            all_loss.append(loss.item())

            tmp_out = out.detach().cpu()
            preds = torch.argmax(tmp_out, 1)
            # print(label,label.shape)
            predict = softmax(tmp_out)

            all_out += predict.cpu().numpy().tolist()
            all_score += preds.numpy().tolist()
            all_label += label.detach().cpu().tolist()
            all_label_single +=label_single.detach().cpu().tolist()
            if config.tensorboard == 1:       
                writer.add_scalars('test_loss', {'test_loss': loss.item()}, index + valid_len * epoch)
            valid_tqdm.set_description('Loss: %.3f,' % (np.mean(all_loss)))
    if config.tensorboard == 1:       
        writer.add_scalars('test_acc_av', {'acc_a': sum(acc_a) / sum(num) * 100}, epoch + 1)
        writer.add_scalars('test_acc_av', {'acc_v': sum(acc_v) / sum(num) * 100}, epoch + 1)
    auc = roc_auc_score(all_label, all_out, multi_class='ovo') * 100
    acc = accuracy_score(all_label_single, all_score) * 100
    AP = []
    all_label = np.array(all_label)
    all_out = np.array(all_out)
    for i in range(len(all_label[0])):
        AP.append(average_precision_score(all_label[:,i],all_out[:,i]))
    f1 = f1_score(all_label_single, all_score, average='macro') * 100

    return {
        'loss': np.mean(all_loss),
        'a_acc': sum(acc_a) / sum(num) * 100,
        'v_acc': sum(acc_v) / sum(num) * 100,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'mAP': np.mean(AP) * 100
    }

def create_folder_if_not_exists(path):
    """
    Create a folder if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def train_test(config, model, model_path, train_loader, valid_loader):
    """
    Train and test the model, and save the results.

    Parameters:
        config (object): Configuration object containing training and model parameters.
        model (nn.Module): The neural network model to be trained and evaluated.
        model_path (str): Path to save the model.
        train_loader (DataLoader): DataLoader for the training set.
        valid_loader (DataLoader): DataLoader for the validation set.
    """
    # Initialize optimizer and learning rate scheduler
    if config.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.patience, 0.1)
    elif config.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.epoch, 0.1)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.StepLR(optimizer, config.epoch, 0.1)

    base_criterion = nn.CrossEntropyLoss()
    criterion = base_criterion

    # Load pre-trained model and evaluate if applicable
    if config.pre_train and os.path.exists(model_path):
        print('Loading pre-trained model...')
        model = torch.load(model_path)
        valid_result = evaluate(config, model, base_criterion, optimizer, valid_loader, 0)
        best_acc = valid_result['acc']
        print(f'Best ACC set to {best_acc}')
    else:
        print('Best ACC set to 0')
        best_acc = 0

    current_date = datetime.now()
    
    # 创建文件夹名为月份加日期，例如：11.22
    folder_name = current_date.strftime('%m.%d')
    
    # 创建文件夹
    
    result_path = 'results/{}/{}'.format(config.dataset, folder_name)
    create_folder_if_not_exists(result_path)

    result_save_path = f'{result_path}/{config.our_model}_seed{config.random_seed}.tsv'
    
    with open(result_save_path, 'a+', newline='', encoding='utf-8') as tsv_f:
        tsv_w = csv.writer(tsv_f, delimiter='\t')
        tsv_w.writerow(['epoch', 'train_loss', 'train_acc', 
                        'test_loss', 'a_acc', 'v_acc', 'test_acc', 'test_auc', 'test_f1', 'test_mAP', 'Train_Time'])
    
    for epoch in range(config.epoch):

        model.zero_grad()
        start_time = time.time()
        train_loss, train_acc = train_epoch(config, model, criterion, optimizer, train_loader,scheduler,epoch)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time: {training_time} seconds")
        valid_result = evaluate(config, model, base_criterion, optimizer, valid_loader, epoch)
        if config.tensorboard == 1:
            writer.add_scalars('loss', {'train_loss': train_loss, 'test_loss': valid_result['loss']}, epoch + 1)
            writer.add_scalars('Acc', {'train_acc': train_acc, 'test_acc': valid_result['acc']}, epoch + 1)

        print(f'TrainACC: {train_acc}, Test Metrics: {valid_result["acc"]}, {valid_result["auc"]}, {valid_result["f1"]}, {valid_result["mAP"]}')
       
        with open(result_save_path, 'a+', newline='', encoding='utf-8') as tsv_f:
            tsv_w = csv.writer(tsv_f, delimiter='\t')
            tsv_w.writerow([epoch, train_loss, train_acc, 
                            valid_result['loss'], valid_result['a_acc'], valid_result['v_acc'], 
                            valid_result['acc'], valid_result['auc'], valid_result['f1'], valid_result['mAP'], training_time])
            
        if valid_result['acc']>best_acc:
            best_acc=valid_result['acc']
            torch.save(model, model_path)
    if config.tensorboard == 1:       
        writer.close()

        
def run(config):

    """
    Main function to run the training and evaluation process for the model.

    Parameters:
        config (object): Configuration object containing training and model parameters.
    """
    set_seed(config.random_seed)  # Set random seed for reproducibility
    config.n_gpu = torch.cuda.device_count()  # Count available GPUs
    print(f"Num GPUs: {config.n_gpu}")
    print(f"Device: {config.device}")

    # Get data loaders for training and testing
    train_loader, test_loader, _ = get_VA_data_loaders(config)

    # Determine the model path and dataset-specific configurations
    models_name = config.our_model
    models_path = os.path.join('ckpt', config.dataset, f"{models_name}.pt")

    # Dataset-specific configurations
    dataset_config = {
        'VGGSound': (309, 3),
        'KS': (31, 3),
        'AVE': (28, 4),
        'CREMA': (6, 1)
    }
    if config.dataset in dataset_config:
        config.n_classes, config.fps = dataset_config[config.dataset]
    else:
        raise NotImplementedError(f'Incorrect dataset name: {config.dataset}')

    # Adjust FPS for fusion method
    if config.fusion_method == 'mmtm':
        config.fps = 1

    print('Current modal: audio & video')

    if config.fusion_method == 'mmtm':
        model = MMTMClassifier(config)
    else:
        model = AVClassifier(config, mask_model=config.mask_model)

    if config.single_pretrain == 1:
        print('Using Pre-Trained Models')
        audio_model = torch.load(models_path[0])
        video_model = torch.load(models_path[1])
        model.audio_net = audio_model.audio_net
        model.video_net = video_model.video_net
    else:
        print('Not Using Pre-Trained Models')
        
    model_path = models_path
# Print configuration details
    print('=' * 200)
    print(f'Begin Training {config.our_model}')
    print(f"GPU: {config.gpu_id} | Dataset: {config.dataset} | Classes: {config.n_classes} | "
          f"Batch Size: {config.batch_size} | Learning Rate: {config.lr} | Epochs: {config.epoch} | "
          f"Model Name: {config.our_model} | Sample Mode: {config.sample_mode} | GN Mode: {config.gn_mode}")
    print('=' * 200)

    model.cuda()  # Move model to GPU
    if config.cal_complex:
        print('---caculate model complex---')
        vdata_bert = torch.randn(64, 1, 257, 1004)  # batch_size=32, sequence_length=128
        vdata_img = torch.randn(64, 3, 3, 224, 224) 
        flops = FlopCountAnalysis(model, (vdata_bert.cuda(), vdata_img.cuda()))
        params = parameter_count(model)
        print("FLOPs %.2f M:" % (flops.total()/ 1000000.0))
        print("Params %.2f M: " %(params['']/ 1000000.0))
        exit()
    # If only testing, evaluate the model and save results

    if config.only_test:
        print('Testing Only')
        base_criterion = nn.CrossEntropyLoss()
        test_result = evaluate(config, model, base_criterion, test_loader)
        
        result_save_path = f'ckpt/{config.dataset}/{config.our_model}_seed{config.random_seed}.tsv'
        with open(result_save_path, 'a+', newline='', encoding='utf-8') as tsv_f:
            tsv_w = csv.writer(tsv_f, delimiter='\t')
            tsv_w.writerow(['Accuracy', 'AUROC', 'Macro F1', 'mAP'])
            print(test_result['acc'], test_result['auc'], test_result['f1'], test_result['mAP'])
            tsv_w.writerow([test_result['acc'], test_result['auc'], test_result['f1'], test_result['mAP']])
        exit()
        
    train_test(config, model, model_path, train_loader, test_loader)

if __name__ == '__main__':
    """
    Main entry point of the script to parse arguments, set configurations, and run the model training/testing.
    """
    parser = argparse.ArgumentParser(description='PyTorch Model Training and Evaluation')
    
    # General configurations
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--data_path', type=str, default='/media/php/data/KS', help='Path to the dataset')
    parser.add_argument('--pre-train', type=int, default=0, help='Whether to use pre-trained models')
    parser.add_argument('--fusion_method', type=str, default='concat', help='Fusion method for multimodal data')
    parser.add_argument('--mask_model', type=int, default=0, help='Mask model option')
    parser.add_argument('--sample_mode', type=str, default='random', choices=['random', 'TopK', 'Adaptive', 'Bernoulli'], help='Sampling mode')
    parser.add_argument('--gn_mode', type=str, default='gn', choices=['gn', 'gn_wn', 'sqrt_gn'], help='Normalization mode')
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature parameter')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer type')
    parser.add_argument('--ratio_random', type=int, default=0, help='Random ratio')
    parser.add_argument('--mask_resnet', type=int, default=1, help='Mask ResNet option')
    parser.add_argument('--mask_ffn', type=int, default=1, help='Mask FFN option')
    parser.add_argument('--isbias', type=int, default=1, help='Whether to use bias')
    parser.add_argument('--bias', type=float, default=0, help='Bias value')
    parser.add_argument('--coeff', type=str, default='balance_av', help='Coefficient type')
    parser.add_argument('--coeff_a', type=float, default=0.5, help='Coefficient for audio modality')
    parser.add_argument('--coeff_v', type=float, default=0.5, help='Coefficient for visual modality')
    parser.add_argument('--fix_ratio', type=int, default=0, help='Whether to fix ratio')
    parser.add_argument('--warm_up_epoch', type=int, default=0, help='Number of warm-up epochs')
    parser.add_argument('--epoch', type=int, default=50, help='Total number of epochs')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--fps', type=int, default=3, help='Frames per second')
    parser.add_argument('--patience', type=int, default=40, help='Patience for learning rate scheduler')
    parser.add_argument('--dataset', type=str, default='KS', help='Name of the dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=64, help='Testing batch size')
    parser.add_argument('--n_classes', type=int, default=31, help='Number of classes')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--modal', type=str, default='multi', help='Modal type')
    parser.add_argument('--param_mode', type=str, default='dynamic', help='Parameter mode')

    # Model configuration options
    parser.add_argument('--single_pretrain', type=int, default=0, help='Use single pre-trained model')
    parser.add_argument('--our_model', type=str, default='SEBM', help='Name of the model')
    parser.add_argument('--only_test', type=int, default=0, help='Whether to only test the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--gpu_id', type=str, default='0', help='ID(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--tensorboard', type=int, default=0, help='Weather save by tensorboard')
    parser.add_argument('--cal_complex', type=int, default=0, help='Parameter mode for dynamic adjustments.')

    # Parse arguments
    config = parser.parse_args()

    modal_v = [0]*config.epoch
    modal_a = [0]*config.epoch

    is_pretrain = 'Pretrain' if config.single_pretrain else 'NoPretrain'
    use_ffn = 'FFN' if config.mask_ffn else 'NoFFN'
    isbias = 'AMSS+' if config.isbias == 1 else 'AMSS'
    if config.mask_ffn == 0 and config.mask_resnet == 0:
        isbias = 'Base'
    
    if config.param_mode == 'fix':
        if config.fix_ratio == 1:
            config.our_model = f"A{config.coeff_a}_V{config.coeff_v}_{config.lr}"
    else:
        config.our_model = f"{isbias}_lr{config.lr}_T{config.temperature}_{config.optimizer}_{config.fusion_method}_{is_pretrain}_{use_ffn}_{config.sample_mode}_{config.gn_mode}_{config.patience}_{config.our_model}"
    # Define target directory for saving results

    # Initialize TensorBoard writer
    if config.tensorboard == 1:
        target_directory = f"{config.dataset}/{config.our_model}{config.coeff}"
        writer = SummaryWriter(target_directory)
        if os.path.exists(target_directory):
            if os.path.isdir(target_directory):
                shutil.rmtree(target_directory)
                print("Directory successfully removed.")
            else:
                print("The target is not a directory.")
        else:
            print("Target directory does not exist.")
    # Set CUDA environment variables
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    # Run the training or testing
    run(config)

    # test(config)

