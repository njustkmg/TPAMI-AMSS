# AMSS/AMSS+
Title: Learning to Rebalance Multi-Modal Optimization by Adaptively Masking Subnetworks

# Data Preparation

You can download the corresponding raw data from the link below and prepare the data according the instructions of the cited paper.

Original Dataset : [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset),[Sarcasm](https://github.com/feiLinX/Multi-modal-Sarcasm-Detection),[Twitter15](https://github.com/jefferyYu/TomBERT),[NVGresutre](https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d)
# Pre-processing
The data processing details are in [OGE-GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022/tree/main) 
# Training

## Audio-Video 

##### Kinetics-Sound

*#AMSS*

python train_test.py --gpu_id=0 --single_pretrain=0 --mask_resnet=1 --mask_ffn=1 --isbias=0 --optimizer=SGD --sample_mode=Adaptive --bias=0 --our_model=balance --patience=60 --epoch=80 --dataset=KS --lr=0.01

*#AMSS+*

python train_test.py --gpu_id=0 --single_pretrain=0 --mask_resnet=1 --mask_ffn=1 --isbias=1  --dataset=KS --optimizer=SGD --sample_mode=Adaptive --bias=0 --our_model=balance --patience=60 --epoch=80 --lr=0.01

##### CREMA-D

#AMSS

python train_test.py --gpu_id=0 --single_pretrain=0 --mask_resnet=1 --mask_ffn=1 --isbias=0 --data_path='/media/php/data/CREMA' --sample_mode=Adaptive --our_model=normal --patience=60 --epoch=80 --lr=0.01 --dataset=CREMA --optimizer='SGD'

#AMSS+

python train_test.py --gpu_id=0 --single_pretrain=0 --mask_resnet=1 --mask_ffn=1 --isbias=1 --data_path='/media/php/data/CREMA' --sample_mode=Adaptive --our_model=normal --patience=60 --epoch=80 --lr=0.01 --dataset=CREMA --optimizer='SGD'

## Text-Img

##### Sarcasm Detection

#AMSS

python -W ignore train_test_IT.py --dataset=Sarcasm --batch_size=32 --test_batch_size=32  --fusion_method=concat --gpu_id=0 --mask_resnet=1 --mask_ffn=1 --isbias=0 --temperature=0.5 --sample_mode=Adaptive --optimizer=Adam --lr=0.000002  --bias=0.2 --gn_mode=gn --patience=40 --epoch=50 --our_model=Adaptive

#AMSS+

python -W ignore train_test_IT.py --dataset=Sarcasm --batch_size=32 --test_batch_size=32  --fusion_method=concat --gpu_id=0 --mask_resnet=1 --mask_ffn=1 --isbias=1 --temperature=0.5 --sample_mode=Adaptive --optimizer=Adam --lr=0.000002  --bias=0.2 --gn_mode=gn --patience=40 --epoch=50--our_model=Adaptive

##### Twitter-15

#AMSS

python -W ignore train_test_IT.py --dataset=Twitter15 --batch_size=16 --test_batch_size=32  --fusion_method=concat --gpu_id=0 --mask_resnet=1 --mask_ffn=1 --isbias=0 --temperature=0.5 --sample_mode=Adaptive --optimizer=Adam --lr=0.000002  --bias=0.2 --gn_mode=gn --patience=40 --epoch=50 --our_model=Adaptive

#AMSS+

python -W ignore train_test_IT.py --dataset=Sarcasm --batch_size=32 --test_batch_size=32  --fusion_method=concat --gpu_id=0 --mask_resnet=1 --mask_ffn=1 --isbias=1 --temperature=0.5 --sample_mode=Adaptive --optimizer=Adam --lr=0.000002  --bias=0.2 --gn_mode=gn --patience=40 --epoch=50--our_model=Adaptive

## RGB+OF+DEPTH

##### NVGesture

#AMSS

python -W ignore train_nv.py --lr=0.01 --epoch=100 --single_pretrain=0 --patience=80 --our_model='meta' --batch-size=4 --mask_resnet=1 --mask_ffn=1 --isbias=0

#AMSS+

python -W ignore train_nv.py --lr=0.01 --epoch=100 --single_pretrain=0 --patience=80 --our_model='meta' --batch-size=4 --mask_resnet=1 --mask_ffn=1 --isbias=1

