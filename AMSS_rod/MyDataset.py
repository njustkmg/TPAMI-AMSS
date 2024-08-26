
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
import random
import torch
import torchvision.transforms.functional as F


resize_transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)),transforms.CenterCrop((224, 224)), transforms.ToTensor()])

class StaticCrop:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    def __call__(self, frame):
        return frame[:, self.x:self.x + self.size,self.y:self.y + self.size]

def load_split_nvgesture(file_with_split = './nvgesture_train_correct_v2.lst',new_path='/media/php/data/nvGesture/Video_data'):
    list_split = list()
    params_dictionary = dict()
    # print(file_with_split)
    with open(file_with_split,'r') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]
        #   print(dict_name)
          for line in f:
            # print(line)
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                parsed = param.split(':')
                key = parsed[0]
                if key == 'label':
                    # make label start from 0
                    label = int(parsed[1]) - 1 
                    params_dictionary['label'] = label
                elif key in ('depth','color','duo_left'):
                    #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                    sensor_name = key
                    #first store path
                    params_dictionary[key] = path + '/' + parsed[1]
                    #store start frame
                    params_dictionary[key+'_start'] = int(parsed[2])
                    params_dictionary[key+'_end'] = int(parsed[3])
            # print(params_dictionary)
              
            for key,value in params_dictionary.items():
                if "./Video_data" in str(value):
                    params_dictionary[key] = value.replace("./Video_data", new_path)
            list_split.append(params_dictionary)
 
    return list_split

def load_data_from_file(example_config, sensor,image_width, image_height):

    path = example_config[sensor] + ".avi"
    start_frame = example_config[sensor+'_start']
    end_frame = example_config[sensor+'_end']
    label = example_config['label']

    frames_to_load = range(start_frame, end_frame)

    chnum = 3 if sensor == "color" else 1
    
    video_container = np.zeros((image_height, image_width, chnum, 80), dtype = np.uint8)

    cap = cv2.VideoCapture(path)

    ret = 1
    frNum = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for indx, frameIndx in enumerate(frames_to_load):    
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(image_width, image_height))
            
            if sensor != "color":
                frame = frame[...,0]
                frame = frame[...,np.newaxis]
            video_container[..., indx] = frame
        else:
            print ("Could not load frame")
            
    cap.release()

    return video_container, label

def load_of_data(index, mode, x, y, start_frame,random_number, new_path='/data/php/nvGesture/Video_data'):

    npy_file_path = '{}/flow_data/{}/{}.npy'.format(new_path,mode,index) 
    loaded_array = np.load(npy_file_path)
    modal_frame = torch.tensor(loaded_array)
    # modal_frame = torch.tensor(loaded_array, dtype=torch.float32)
    modal_frame = modal_frame.permute(3, 2, 0, 1)
    if mode=='train':
        # modal_frame = modal_frame[start_frame:start_frame + 64,:, :, :]
        # transform_train = transforms.Compose([transforms.RandomHorizontalFlip(random_number)])
        modal_frame_resized = torch.stack([frame[:,y:y+224, x:x+224] for frame in modal_frame], dim=0)
        if random_number:
            modal_frame_resized = torch.flip(modal_frame_resized, dims=(3,))
        # modal_frame_resized =torch.stack([transform_train(frame) for frame in modal_frame_resized_st], dim=0)
    else:
        modal_frame_resized = torch.stack([frame[:,16:16+224,16:16+224] for frame in modal_frame], dim=0)

    # modal_frame_resized = modal_frame_resized[:79, :, :, :]
    # print(modal_frame_resized.shape)
    # if mode=='train':
    #     modal_frame_resized_st = torch.stack([resize_transform(frame) for frame in modal_frame], dim=0)
    # else:
    #     # print(modal_frame.shape)
    # modal_frame_resized = torch.stack([resize_transform_test(frame) for frame in modal_frame], dim=0)

    return  modal_frame_resized
    
def process_frames(modal_frame, x ,y,start_frame, random_number, mode = 'train', modal = 'rgb'):
    # Ensure both datasets have matching starting frame indices

    # Randomly select 64 consecutive frames or zero-pad if needed

    if mode == 'train':
        video_frames = modal_frame.shape[-1]  # Get the number of frames
        if video_frames < 64:
            print('Lower Than 64 frams')
            # Zero-pad on both sides to obtain 64 frames
            padding = 64 - video_frames
            left_padding = random.randint(0, padding)
            right_padding = padding - left_padding
            modal_frame = np.pad(modal_frame, ((0, 0), (0, 0), (0, 0), (left_padding, right_padding)), mode='constant')
        else:
            
            modal_frame = modal_frame[:, :, :, start_frame:start_frame + 64]
        modal_frame = torch.tensor(modal_frame)
        # modal_frame = torch.tensor(modal_frame, dtype=torch.float32)
        modal_frame = modal_frame.permute(3, 2, 0, 1)               
        modal_frame_resized_nd = torch.stack([frame[:,y:y+224, x:x+224] for frame in modal_frame], dim=0)
        if modal!='depth':
            transform_train = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(random_number),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])            
            modal_frame_resized = torch.stack([transform_train(frame) for frame in modal_frame_resized_nd], dim=0)
            # print('After:' ,modal_frame_resized)

        else:
            transform_train = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(random_number),transforms.ToTensor()])
            modal_frame_resized = torch.stack([transform_train(frame) for frame in modal_frame_resized_nd], dim=0)

    # Resize frames to 256x256
    else:
        modal_frame = torch.tensor(modal_frame)
        modal_frame = modal_frame.permute(3, 2, 0, 1)
        # print(modal_frame.shape)
        if modal!='depth':
            resize_transform_test = transforms.Compose([transforms.ToPILImage(),\
                                    transforms.CenterCrop((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            modal_frame_resized = torch.stack([resize_transform_test(frame) for frame in modal_frame], dim=0)
        else:
            resize_transform_test = transforms.Compose([transforms.ToPILImage(),\
                                    transforms.CenterCrop((224, 224)), transforms.ToTensor()])
            modal_frame_resized = torch.stack([resize_transform_test(frame) for frame in modal_frame], dim=0)

    return modal_frame_resized

class NvGestureDataset(Dataset):
    def __init__(self,config, mode='train'):
        self.config = config
        self.sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
        self.file_lists = dict()
        self.file_lists["train"] = config.data_path + "nvgesture_train_correct_v2.lst"
        self.file_lists["test"] = config.data_path + "nvgesture_test_correct_v2.lst"
        self.train_list = load_split_nvgesture(file_with_split = self.file_lists["train"], new_path = config.data_path + 'Video_data')
        self.test_list  = load_split_nvgesture(file_with_split = self.file_lists["test"], new_path = config.data_path + 'Video_data')
        self.mode = mode
        if mode == 'train':
            self.data_list = self.train_list
        else:
            self.data_list = self.test_list

        # print(len(train_list))
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        x = random.randint(0, 256 - 224)
        y = random.randint(0, 256 - 224)

        start_frame = random.randint(0, 80 - 64)
        # random_seed = random.choice([0, 1])
        random_seed = 0
        rgb, label = load_data_from_file(example_config = self.data_list[idx], sensor = self.sensors[0], image_width = 256, image_height = 256)
        rgb = process_frames(rgb, x, y, start_frame, random_seed, self.mode, 'rgb')
        rgb = rgb.permute(1,0,2,3)
        
        depth,_ = load_data_from_file(example_config = self.data_list[idx], sensor = self.sensors[1], image_width = 256, image_height = 256)
        depth = process_frames(depth, x, y,start_frame, random_seed, self.mode,'depth')
        depth = depth.permute(1,0,2,3)
        # print(self.mode)
        # print(depth)
        # exit()
        of = load_of_data(idx, self.mode, x, y,start_frame, random_seed, new_path = self.config.data_path + 'Video_data')
        of = of.permute(1,0,2,3)
 
        # print(rgb)

        # print(rgb.shape)

        one_hot = np.eye(self.config.n_classes)
        one_hot_label = one_hot[label]
        label = torch.FloatTensor(one_hot_label)
        # return rgb, of, depth,label
        return rgb, of, depth,label
    
