import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
import os
import torch
from torchvision import transforms
from typing import Tuple
import csv
import torchvision
from torch import Tensor
import librosa

class MultiModalDataset(Dataset):
    def __init__(self, data_path, img_path, tokenizer,config,train_transform):
        self.data = pd.read_csv(data_path,sep='\t')
        # print("Column names:", self.data.columns)
        # print(self.data.head())
        print(f'Read {len(self.data)} images from file {data_path.split("/")[-1]}.')
        self.train_transform= train_transform
        self.labels = self.data['Label']
        self.contents = self.data['String']
        self.imgs =  self.data['ImageID']
        self.img_dir = img_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.errors = 0

    def __len__(self):
        return len(self.labels)
    def get_tokenized(self,text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0]*(self.max_seq_len-len(encode_result))
        encode_result += padding
        return torch.tensor(encode_result)


    def __getitem__(self, index):
        image = Image.open(
            os.path.join(self.img_dir, self.imgs[index])
        ).convert("RGB")
        image = self.train_transform(image)

        sentence = self.get_tokenized(self.contents[index])

        one_hot = np.eye(self.config.n_classes)
        one_hot_label = one_hot[self.labels[index]]
        label = torch.FloatTensor(one_hot_label)
        
        return sentence,image,label

    def __len__(self):
        return len(self.labels)
