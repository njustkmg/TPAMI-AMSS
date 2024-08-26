
from MyDataset import NvGestureDataset
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer
# import monai.transforms as transforms_v1
import os
from torchvision import transforms

def get_ROD_data_loaders(config):

    if config.dataset == 'nvGesture':           
        train_set = NvGestureDataset(config, mode='train')
        test_set = NvGestureDataset(config, mode='test')
    else:
        raise ValueError('no dataset')
    
    train_loader = DataLoader(
        train_set,
        # test_set,
        batch_size=config.batch_size,
        sampler= None,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader,test_loader


