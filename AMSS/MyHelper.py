
from MyDataset import VADataset,CramedDataset,AVEDataset
from torch.utils.data import DataLoader
# from pytorch_pretrained_bert import BertTokenizer
import os
from torchvision import transforms


def get_VA_data_loaders(config):
    if config.dataset == 'CREMA':        
        train_set = CramedDataset(config,mode='train')
        test_set = CramedDataset(config,mode='test')
        weight_set = test_set
    elif config.dataset == 'KS':
        train_set = VADataset(config,mode='train')
        # weight_set = VADataset(config,mode='weight')
        test_set = VADataset(config,mode='test')
    elif config.dataset == 'AVE':
        train_set = AVEDataset(config,mode='train')
        test_set = AVEDataset(config,mode='test')
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
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader,test_loader,test_loader


