from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *

class MaskDataset(Dataset) :
    def __init__(self, path_list, label_list, age_list, gender_list, mask_list, transform, aug_transform = None, need = False) :
        self.X = []
        self.y = []
        self.transform = transform
        self.aug_transform = aug_transform
        
        '''
         데이터 불균형 문제를 해결하고자 age, gender, mask에 따라 data augmentation적용 횟수를 다르게 해주었다. 
         이때 need(train, test)에 따라서 data augmentation을 적용 여부가 달라진다.
        '''
        for path, label, age, gender, mask in tqdm(zip(path_list, label_list, age_list, gender_list, mask_list)) :
            image = Image.open(path)
            self.X.append(image)
            self.y.append(label)
            if need :
                if 21 <= age <= 26 and gender == 'male' and len(mask) == 5 :
                    for _ in range(1) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 21 <= age <= 26 and gender == 'male' :
                    for _ in range(10) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 21 <= age <= 26 and gender == 'female' and len(mask) == 5 :
                    for _ in range(1) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 21 <= age <= 26 and gender == 'female' :
                    for _ in range(10) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 27 <= age <= 47 and gender == 'male' and len(mask) == 5 :
                    for _ in range(5) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 27 <= age <= 47 and gender == 'male' :
                    for _ in range(8) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 27 <= age <= 47 and gender == 'female' and len(mask) == 5:
                    for _ in range(5) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 27 <= age <= 47 and gender == 'female' :
                    for _ in range(8) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 48 <= age <= 54 and gender == 'male' and len(mask) == 5 :
                    for _ in range(2) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 48 <= age <= 54 and gender == 'male' :
                    for _ in range(2) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 48 <= age <= 54 and gender == 'female' and len(mask) == 5:
                    for _ in range(2) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
                elif 48 <= age <= 54 and gender == 'female' :
                    for _ in range(2) :
                        self.y.append(label)
                        self.X.append(Image.fromarray(self.aug_transform(image = np.array(image))['image']))
    
    def __getitem__(self, idx) :
        X = self.X[idx]
        y = self.y[idx]
        X = self.transform(image = np.array(X))['image']
        return torch.tensor(X, dtype = torch.float), torch.tensor(y, dtype = torch.long)
        
    def __len__(self) :
        len_dataset = len(self.X)
        return len_dataset

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
