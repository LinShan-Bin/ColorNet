import os
import fnmatch
import json
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


DATA_PATH = './dataset/medium/'
RESOLUTION = (224, 224)

class ColorfulClothes(Dataset):
    def __init__(self, data_path, train):
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
            images.append(read_image(image))
        label = self.labels[self.image_folds[index]]
        return images, label
    
    
class ColorfulClothesUnfolded(Dataset):
    def __init__(self, data_path, train):
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
        image = read_image(image_path)
        optional_tags = self.labels[self.product_id[index]]['optional_tags']
        label = self.labels[self.product_id[index]]['imgs_tags'][id][self.image_name[index]]
        return image, optional_tags, label
    
if __name__ == '__main__':
    train_dataset = ColorfulClothes(DATA_PATH, True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, (image, label) in enumerate(train_loader):
        pprint(image)
        pprint(label)
        if i == 0:
            break
        
    unfolded = ColorfulClothesUnfolded(DATA_PATH, True)
    loader = DataLoader(unfolded, batch_size=1, shuffle=True)
    for i, (image, optional_tags, label) in enumerate(loader):
        pprint(image)
        pprint(optional_tags)
        pprint(label)
        if i == 0:
            break
        