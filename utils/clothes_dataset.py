from copy import copy
import os
import fnmatch
import json
from pprint import pprint
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize


DATA_PATH = './dataset/medium/'
RESOLUTION = (224, 224)

class ColorfulClothes(Dataset):
    def __init__(self, data_path, train, resolution=(224, 224)):
        self.resolution = resolution
        self.tranform = torchvision.transforms.Compose([
            Resize(resolution),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if train:
            label_path = os.path.join(data_path, 'train_all.json')
            self.data_path = os.path.join(data_path, 'train')
        else:
            label_path = os.path.join(data_path, 'test_all.json')
            self.data_path = os.path.join(data_path, 'test')
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        self.image_folds = list(self.labels.keys())
        
    def __len__(self):
        return len(self.image_folds)
    
    def __getitem__(self, index):
        fold_path = os.path.join(self.data_path, self.image_folds[index])
        image_path = fnmatch.filter(os.listdir(fold_path), '*.jpg')
        images = []
        for i in range(len(image_path)):
            image = os.path.join(fold_path, image_path[i])
            image = read_image(image).float() / 255.
            image = self.tranform(image)
            images.append(read_image(image))
        label = self.labels[self.image_folds[index]]
        return images, label
    
    
class ColorfulClothesUnfolded(Dataset):
    def __init__(self, data_path, train, resolution=(224, 224)):
        self.resolution = resolution
        self.tranform = torchvision.transforms.Compose([
            Resize(resolution),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
    def __len__(self):
        return len(self.image_name)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.product_id[index], self.image_name[index])
        image_id = self.image_name[index].split('.')[0]
        id = int(image_id.split('_')[-1])
        image = read_image(image_path).float() / 255.
        image = self.tranform(image)
        optional_tags = self.labels[self.product_id[index]]['optional_tags']
        label = self.labels[self.product_id[index]]['imgs_tags'][id][self.image_name[index]]
        return image, optional_tags, label


# TODO: Optimize performance.
class ColorfulClothesBin(Dataset):
    def __init__(self, data_path, embed, train, resolution=(224, 224)):
        self.resolution = resolution
        self.tranform = torchvision.transforms.Compose([
            Resize(resolution),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
        image_path = os.path.join(self.data_path, self.product_id[index], self.image_name[index])
        image_id = self.image_name[index].split('.')[0]
        id = int(image_id.split('_')[-1])
        org_img = self.tranform(read_image(image_path))
        image = org_img.float() / 255.
        image = self.tranform(image)
        optional_tags = self.labels[self.product_id[index]]['optional_tags']
        true_tag = self.labels[self.product_id[index]]['imgs_tags'][id][self.image_name[index]]
        label = torch.randint(0, 2, (1,)) if (len(optional_tags) != 1) else torch.tensor([1])
        if label == 1:
            tag = true_tag
        else:
            optional_tags = copy(optional_tags)
            optional_tags.remove(true_tag)
            tag = optional_tags[torch.randint(0, len(optional_tags), (1,))]
        etag = self.embed(tag)
        return org_img, image.float(), etag.float(), label.float()


class ColorfulClothesCLF(Dataset):
    def __init__(self, data_path, class_num, embed, train, resolution=(224, 224)):
        self.resolution = resolution
        self.tranform = torchvision.transforms.Compose([
            Resize(resolution),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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
        image_path = os.path.join(self.data_path, self.product_id[index], self.image_name[index])
        image_id = self.image_name[index].split('.')[0]
        id = int(image_id.split('_')[-1])
        image = self.tranform(read_image(image_path))
        image = image.float() / 255.
        
        optional_tags = self.labels[self.product_id[index]]['optional_tags']
        true_tag = self.labels[self.product_id[index]]['imgs_tags'][id][self.image_name[index]]
        
        optional_labels = [self.embed(tag) for tag in optional_tags]
        optional_labels = optional_labels + [-1] * (self.class_num - len(optional_labels))
        optional_labels = torch.tensor(optional_labels, dtype=torch.long)
        true_label = self.embed(true_tag)
        true_label = torch.tensor([true_label], dtype=torch.long)
        
        return image, optional_labels, true_label
