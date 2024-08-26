import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Twitter15 Dataset Config')
    # basic settings
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='whether to use gpu')
    parser.add_argument('--random_seed', type=int, default=42,help='random seed')
    parser.add_argument('--data_path', type=str, default='/home/admin1/php/data_processing')
    parser.add_argument('--pre-train', type=int, default=0)
    parser.add_argument('--fusion_method', type=str, default='concat')
    parser.add_argument('--mask_model', type=int, default=0)
    parser.add_argument('--log_name', type=str, default='ffn_mask')
    parser.add_argument('--sample_mode', type=str, default='random',choices=['random','GN','Adaptive','Bernoulli'])
    parser.add_argument('--gn_mode', type=str, default='gn',choices=['gn','gn_wn','sqrt_gn'])
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--ratio_random', type=int, default=0)
    parser.add_argument('--mask_ratio', type=float, default=1)
    parser.add_argument('--mask_resnet', type=int, default=1)
    parser.add_argument('--mask_ffn', type=int, default=1)
    parser.add_argument('--isbias', type=int, default=1)
    parser.add_argument('--bias', type=float, default=0)


    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--coeff', type=str, default='balance_av')
    parser.add_argument('--gimma', type=int, default=1)
    parser.add_argument('--balanceA', type=int, default=1)
    parser.add_argument('--balanceV', type=int, default=1)
    parser.add_argument('--ratio_weight', type=float, default=0.5)
    parser.add_argument('--keep_ratio', type=float, default=0.9)
    parser.add_argument('--clf_weight', type=float, default=1)
    
    parser.add_argument('--warm_up_epoch', type=int, default=0,
                        help='the total number of epoch in the training')
    parser.add_argument('--epoch', type=int, default=50,
                        help='the total number of epoch in the training')
    parser.add_argument('--num_workers', type=int , default=8)
    parser.add_argument('--visual_lr', type=float, default=0.001)
    parser.add_argument('--audio_lr', type=float, default=0.001)
    parser.add_argument('--multi_lr', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay_visual', type=float, default=0.0001)
    parser.add_argument('--weight_decay_audio', type=float, default=0.0001)
    parser.add_argument('--weight_decay_multi', type=float, default=0.0001)

    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--dataset',default='KS',type=str)
    parser.add_argument('--batch-size', default=32, type=int,help='train batchsize')
    
    parser.add_argument('--single_epoch', default=30, type=int)
    parser.add_argument('--test-batch-size', default=64, type=int,help='train batchsize')
    parser.add_argument('--n_classes', type=int, default=31)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--modal', type=str, default='DER_av_balance_v')
    parser.add_argument('--single_pre_train',type=int,default=0,
                        help='单模态已经预训练好，控制使模型使用单模态预训练好的特征提取器和分类头')
    parser.add_argument("--single_pretrain", default=0,type=int,
                        help="visual pre-train path")
    parser.add_argument('--visual_pretrain', type=str, default="video")
    parser.add_argument("--audio_pretrain", default="audio",type=str,
                        help="audio pre-train path")
    parser.add_argument("--our_model", default="SEBM",type=str,
                    help="Our Model's Name")
    parser.add_argument("--only_test",type=int,default=0)
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    return parser






