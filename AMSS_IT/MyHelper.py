
from MyDataset import MultiModalDataset
from torch.utils.data import DataLoader
from pytorch_pretrained_bert import BertTokenizer
import os
from torchvision import transforms

def get_IT_datasets(config):

    if config.dataset=='Fakeddit':
        data_mean = [0.432, 0.4572, 0.4912]
        data_std = [0.2316, 0.2340, 0.2426]
    elif config.dataset=='Twitter15':
        data_mean = [0.42455, 0.44051, 0.47076]
        data_std = [0.24485, 0.24617, 0.25451]
    elif config.dataset=='IMDB':
        data_mean = [0.4104, 0.4467, 0.5012]
        data_std = [0.2722, 0.2829, 0.2953]
    elif config.dataset=='Twitter17':
        data_mean = [0.42397, 0.4301, 0.4711]
        data_std = [0.25305, 0.2558, 0.2686]
    elif config.dataset=='HatefulMemes':
        data_mean = [0.45963, 0.48171, 0.5127]
        data_std = [0.2821, 0.27746, 0.28142]    
    elif config.dataset=='CrisisMMD':
        data_mean = [0.452274, 0.469894, 0.485588]
        data_std =  [0.24234, 0.240737, 0.24751]
    else:
        data_mean = [0.48456249428840364, 0.4988762843596956, 0.530657871261656]
        data_std = [0.24398715912049204, 0.24868027118814742, 0.2573621194247526]
    path_temp_train =''
    path_temp_test =''
    path_temp_dev =''

    if config.dataset=='Sarcasm':
        path_temp_train ='images'
        path_temp_test ='images'

    elif config.dataset=='Twitter15':
        path_temp_train = 'twitter2015_images'
        path_temp_test = 'twitter2015_images'

    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean,std=data_std)
])
    
    tokenizer = (BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True))
    
    train_set = MultiModalDataset(
        os.path.join(config.data_path, config.dataset,"annotations",'train.tsv'),
        os.path.join(config.data_path, config.dataset, path_temp_train),
        tokenizer,
        config,
        train_transform,
    )
    
    test_set = MultiModalDataset(
        os.path.join(config.data_path, config.dataset,"annotations", "test.tsv"),
        os.path.join(config.data_path, config.dataset, path_temp_test),
        tokenizer,
        config,
        train_transform
    )

    return train_set,test_set

def get_IT_data_loaders(config):
    # collate = functools.partial(collate_fn_mmbt, args=args)

    train_set,test_set = get_IT_datasets(config)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(labeled_set)
    
    train_loader = DataLoader(
        train_set,
        batch_size= config.batch_size,
        sampler= None,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch_size,
        #         sampler=SequentialSampler(test_set),
        shuffle=False,
        num_workers=config.num_workers,
    )
    # return labeled_loader,test_loader
    return train_loader,test_loader
