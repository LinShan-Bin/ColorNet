import os
import fnmatch
import json
from random import sample
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize, ToTensor


DATA_PATH = './dataset/medium/'
RESOLUTION = (224, 224)


class ColorfulClothesCLF(Dataset):
    def __init__(self, data_path, class_num, embed, train, mask=True, resolution=(224, 224)):
        self.resolution = resolution
        self.tranform = torchvision.transforms.Compose([
            Resize(resolution),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_num = class_num
        self.embed = embed
        self.mask = mask

        if train:
            label_path = os.path.join(data_path, 'train_all.json')
            self.data_path = os.path.join(data_path, 'train')
        else:
            label_path = os.path.join(data_path, 'test_all.json')
            self.data_path = os.path.join(data_path, 'test')
        with open(label_path, 'r') as f:
            self.json = json.load(f)
        image_folds = list(self.json.keys())
        self.image_name = []
        self.product_id = []
        for i in range(len(image_folds)):
            fold_path = os.path.join(self.data_path, image_folds[i])
            names = fnmatch.filter(os.listdir(fold_path), '*.jpg')
            self.image_name += names
            self.product_id += [image_folds[i]] * len(names)
            
        self.length = len(self.image_name)
        
    def clean_and_analyse(self):
        remove_list = []
        data_distribution = torch.zeros(self.class_num, dtype=torch.long)
        self.labels = []
        for index in range(self.length):
            image_id = self.image_name[index].split('.')[0]
            id = int(image_id.split('_')[-1])
            true_tag = self.json[self.product_id[index]]['imgs_tags'][id][self.image_name[index]]
            label = self.embed(true_tag)
            if label is None:
                remove_list.append(index)
            else:
                data_distribution[label] += 1
                self.labels.append(label)
        self.data_distribution = data_distribution
        for index in reversed(remove_list):
            self.image_name.pop(index)
            self.product_id.pop(index)
        self.length = len(self.labels)

        # Multinomial distribution for sampling
        data_distribution = np.array(data_distribution)
        data_prob = data_distribution / data_distribution.sum()
        sample_prob = np.sqrt(data_prob) / np.sum(np.sqrt(data_prob)) / data_prob
        self.sample_weights = []
        for i in range(self.length):
            self.sample_weights.append(sample_prob[self.labels[i]])
        return self.data_distribution
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.product_id[index], self.image_name[index])
        image = self.tranform(Image.open(image_path))
        
        optional_tags = self.json[self.product_id[index]]['optional_tags']
        
        multi_hot = torch.zeros(self.class_num, dtype=torch.float)
        for tag in optional_tags:
            e = self.embed(tag)
            if e is not None:
                multi_hot[e] = 1.
        
        true_label = torch.tensor([self.labels[index]], dtype=torch.long)
        
        if self.mask:
            mask_path = image_path.replace('jpg', 'pt')
            mask = torch.load(mask_path)
        else:
            mask = torch.ones(self.resolution).unsqueeze(0)  # 统一接口方便处理
        
        return image, mask, multi_hot, true_label, self.product_id[index]
    

class ColorfulClothesLabel(Dataset):
    def __init__(self, data_path, class_num, embed, train):
        self.class_num = class_num
        self.embed = embed
        if train:
            label_path = os.path.join(data_path, 'train_all.json')
            self.data_path = os.path.join(data_path, 'train')
        else:
            label_path = os.path.join(data_path, 'test_all.json')
            self.data_path = os.path.join(data_path, 'test')
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        image_folds = list(self.labels.keys())
        self.image_name = []
        self.product_id = []
        for i in range(len(image_folds)):
            fold_path = os.path.join(self.data_path, image_folds[i])
            names = fnmatch.filter(os.listdir(fold_path), '*.jpg')
            self.image_name += names
            self.product_id += [image_folds[i]] * len(names)
            
        self.length = len(self.image_name)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image_id = self.image_name[index].split('.')[0]
        id = int(image_id.split('_')[-1])
        
        true_tag = self.labels[self.product_id[index]]['imgs_tags'][id][self.image_name[index]]
        
        true_label = self.embed(true_tag)
        return true_label


# For Generate mask
class ColorfulClothesIMG(Dataset):
    def __init__(self, data_path, train, org, resolution=(224, 224)):
        self.resolution = resolution
        self.org = org
        self.tranform = torchvision.transforms.Compose([
            Resize(resolution),
            ToTensor()
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Segmentation models do not need normalization.
        ])
        if train:
            label_path = os.path.join(data_path, 'train_all.json')
            self.data_path = os.path.join(data_path, 'train')
        else:
            label_path = os.path.join(data_path, 'test_all.json')
            self.data_path = os.path.join(data_path, 'test')
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        image_folds = list(self.labels.keys())
        self.image_name = []
        self.product_id = []
        for i in range(len(image_folds)):
            fold_path = os.path.join(self.data_path, image_folds[i])
            names = fnmatch.filter(os.listdir(fold_path), '*.jpg')
            self.image_name += names
            self.product_id += [image_folds[i]] * len(names)
            
        self.length = len(self.image_name)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.product_id[index], self.image_name[index])
        image_id = self.image_name[index].split('.')[0]
        id = int(image_id.split('_')[-1])
        org_image = Image.open(image_path)
        image = self.tranform(org_image)
        if self.org:
            return image, image_path, org_image
        else:
            return image, image_path


class ColorfulClothesTest(Dataset):
    def __init__(self, data_path, class_num, embed, mask=True, resolution=(224, 224)):
        self.resolution = resolution
        self.tranform = torchvision.transforms.Compose([
            Resize(resolution),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_num = class_num
        self.mask = mask
        self.embed = embed

        label_path = os.path.join(data_path, 'test_all.json')
        self.data_path = os.path.join(data_path, 'test')
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        image_folds = list(self.labels.keys())
        self.image_name = []
        self.product_id = []
        for i in range(len(image_folds)):
            fold_path = os.path.join(self.data_path, image_folds[i])
            names = fnmatch.filter(os.listdir(fold_path), '*.jpg')
            self.image_name += names
            self.product_id += [image_folds[i]] * len(names)
            
        self.length = len(self.image_name)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.product_id[index], self.image_name[index])
        image_id = self.image_name[index].split('.')[0]
        id = int(image_id.split('_')[-1])
        image = self.tranform(Image.open(image_path))
        
        optional_tags = self.labels[self.product_id[index]]['optional_tags']
        
        multi_hot = torch.zeros(self.class_num, dtype=torch.float)
        for tag in optional_tags:
            e = self.embed(tag)
            for i in e:
                multi_hot[i] = 1.
        
        if self.mask:
            mask_path = image_path.replace('jpg', 'pt')
            mask = torch.load(mask_path)

        else:
            mask = torch.ones(self.resolution).unsqueeze(0)  # 统一接口方便处理
            
        return image, mask, multi_hot, self.product_id[index], id, self.image_name[index]
