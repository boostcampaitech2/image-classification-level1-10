import torch
import torch.nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageEnhance
import os
import copy
from options import Options
import albumentations as A
import albumentations.pytorch as AP
import numpy as np

'''
elastic 
focal-loss,
f1-loss
'''
masksize = 256
gendsize = 256
centsize = 320
n_workers = {'train': 4, 'test': 0}

# IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

baseA = A.Compose([
    A.CoarseDropout(max_holes=3, max_height=40, max_width=40),
    A.CoarseDropout(max_holes=1, max_height=100, max_width=70),
    A.ElasticTransform(),
    A.GaussNoise(),
    A.GridDistortion(),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456)),
    AP.transforms.ToTensorV2(),
])

maskT = copy.deepcopy(baseA)
# maskT.transforms[2].size = (masksize, masksize)
gendT = copy.deepcopy(baseA)
ageT  = copy.deepcopy(baseA)

transform_dict = {'mask': maskT, 'gender': gendT, 'age': ageT}

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def NormalLoader(isTrain, batch_size):
    if isTrain:
        dataset = train_val_dataset(NormalDataset(isTrain))

        dataloader = {x: DataLoader(
            dataset = dataset[x],
            batch_size = batch_size,
            shuffle = True,
            num_workers = n_workers['train'],
            drop_last = isTrain,
        ) for x in ['train', 'val']}

    else:
        dataset = NormalDataset(isTrain)

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = n_workers['test'],
            drop_last = isTrain,
        )
    return dataloader


def ProjectedLoader(name, isTrain, batch_size):
    if isTrain:
        dataset = train_val_dataset(ProjectedDataset(name=name, isTrain=True))

        dataloader = {x: DataLoader(
            dataset = dataset[x],
            batch_size = batch_size,
            shuffle = True,
            num_workers = n_workers['train'],
            drop_last = isTrain,
        ) for x in ['train', 'val']}

    else:
        dataset = ProjectedDataset(name=name, isTrain=False)

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = n_workers['test'],
            drop_last = False,
        )
    return dataloader



class NormalDataset(Dataset):
    def __init__(self, isTrain):
        self.x = []
        self.y = []
        self.dataroot = '/opt/ml/input/purified'
        self.n_class = 18
        self.isTrain = isTrain
        self._get_xy()
        self.cache = {}

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        fname = self.x[idx].split('/')[-1]
        
        if self.isTrain:
            if fname in self.cache:
                X = self.cache[ fname ]
            else:
                X = np.array(Image.open(self.x[idx]))
                X = self._preprocess(X)
                self.cache[fname] = X
            X = baseA(image=X)['image']
            return X, self.y[idx]
        
        else:
            X = np.array(Image.open(self.x[idx]))
            X = self._preprocess(X)
            X = baseA(image=X)['image']
            return X, fname
    
    def _preprocess(self, X: str)->Image:
        X = A.CenterCrop(centsize,centsize)(image=X)['image']
        X = A.Resize(256,256)(image=X)['image']
        # X = ImageEnhance.Contrast(X).enhance(5)
        # X = ImageEnhance.Color(X).enhance(0.8)
        return X

    def _get_xy(self):
        if self.isTrain:
            train_root = os.path.join(self.dataroot, 'train')
            for cls in range(self.n_class):
                img_paths = glob(f'{train_root}/{cls}/**/*.*', recursive=True)
                self.x.extend( img_paths )
                self.y.extend( [cls] * len(img_paths) )
        
        # Test
        else:
            test_root = os.path.join(self.dataroot, 'test')
            self.x = glob(f'{test_root}/*.*')



class ProjectedDataset(Dataset):
    def __init__(self, name, isTrain):
        self.x = []
        self.y = []
        self.dataroot = '/opt/ml/input/purified'
        self.name = name
        self.n_class = 18
        self.isTrain = isTrain
        self._get_xy()
        self.cache = {}

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        fname = self.x[idx].split('/')[-1]
        
        if self.isTrain:
            if fname in self.cache:
                X = self.cache[ fname ]
            else:
                X = np.array(Image.open(self.x[idx]))
                X = self._preprocess(X)
                self.cache[fname] = X
            X = transform_dict[self.name](image=X)['image']
            return X, self.y[idx]
        
        else:
            X = np.array(Image.open(self.x[idx]))
            X = self._preprocess(X)
            X = transform_dict[self.name](image=X)['image']
            return X, fname
    
    def _preprocess(self, X)->Image:
        X = A.CenterCrop(centsize,centsize)(image=X)['image']
        X = A.Resize(256,256)(image=X)['image']

        # X = ImageEnhance.Contrast(X).enhance(5)
        # X = ImageEnhance.Color(X).enhance(0.8)
        return X

    def _get_xy(self):
        if self.isTrain:
            train_root = os.path.join(self.dataroot, 'train')

            if self.name == 'mask':
                grouped_cls = [
                    [0,1,2,3,4,5],
                    [6,7,8,9,10,11],
                    [12,13,14,15,16,17]]
            elif self.name == 'gender':
                grouped_cls = [
                    [0,1,2, 6,7,8, 12,13,14],
                    [3,4,5, 9,10,11, 15,16,17]]
            elif self.name == 'age':
                grouped_cls = [
                    [0,3,6,9,12,15],
                    [1,4,7,10,13,16],
                    [2,5,8,11,14,17]]
            
            else: raise Exception ('[ProjectedDataset]: make sure the name. It should be in (mask | gender | age)')

            for cls in range(self.n_class):
                for i, g_cls in enumerate(grouped_cls):
                    if cls in g_cls: 
                        projected_cls = i
                
                img_paths = glob(f'{train_root}/{cls}/**/*.*', recursive=True)
                self.x.extend( img_paths )
                self.y.extend( [projected_cls] * len(img_paths) )
        
        # Test
        else:
            test_root = os.path.join(self.dataroot, 'test')
            self.x = glob(f'{test_root}/*.*')


if __name__ == '__main__':
    # mask = ProjectedDataset(name='mask', isTrain=True)
    # gend = ProjectedDataset(name='gender', isTrain=True)
    age  = ProjectedDataset(name='age', isTrain=True)

    # mask_test = ProjectedDataset(name='mask', isTrain=False)
    # gend_test = ProjectedDataset(name='gender', isTrain=False)
    # age_test = ProjectedDataset(name='age', isTrain=False)

    for epoch in range(5):
        for idx, (img, label) in enumerate(tqdm(age)):
            if idx == int(len(age)/2):
                print(f'{idx}: {label}')