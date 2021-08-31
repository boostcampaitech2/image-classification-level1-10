from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
import os
import copy
import albumentations as A
import albumentations.pytorch as AP
import numpy as np
from PIL import Image
import random

dataroot = '/opt/ml/input/cropped'
recur = False

#efficientnet_b2 256
basesize = 256
# centsize = 320
n_workers = {'train': 4, 'test': 4}

baseA = A.Compose([
    A.CoarseDropout(max_holes=3, max_height=40, max_width=40),
    A.CoarseDropout(max_holes=1, max_height=100, max_width=70),
    A.ElasticTransform(),
    A.GaussNoise(),
    A.GridDistortion(),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.54892884,0.50471638,0.48014299), std=(0.23508963,0.24486722,0.24449045)),
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



def load_data(isTrain, batch_size, name=None, expand=False):
    if isTrain:
        if name in ['mask', 'gender', 'age']:
            print('[Dataset]:\t ProjectedDataset loaded')
            dataset = train_val_dataset(ProjectedDataset(name=name, isTrain=True))
        elif expand:
            print('[Dataset]:\t NormalDataset_oversampled loaded')
            dataset = train_val_dataset(NormalDataset_oversampled(isTrain))
        else:
            print('[Dataset]:\t NormalDataset loaded')
            dataset = train_val_dataset(NormalDataset(isTrain=True))
        

        dataloader = {x: DataLoader(
            dataset = dataset[x],
            batch_size = batch_size,
            shuffle = True,
            num_workers = n_workers['train'],
            drop_last = True,
        ) for x in ['train', 'val']}

    else:
        if name in ['mask', 'gender', 'age']:
            print('[Dataset]:\t ProjectedDataset for TEST loaded')
            dataset = ProjectedDataset(name=name, isTrain=False)
        else:
            print('[Dataset]:\t NormalDataset for TEST loaded')
            dataset = NormalDataset(isTrain=False)

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = n_workers['test'],
            drop_last = False,
        )
    return dataloader

class NormalDataset(Dataset):
    def __init__(self, isTrain):
        self.x = []
        self.y = []
        self.dataroot = dataroot
        self.n_class = 18
        self.isTrain = isTrain
        self._get_xy()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        fname = self.x[idx].split('/')[-1]
        X = np.array(Image.open(self.x[idx]))
        X = self._preprocess(X)
        
        if self.isTrain:
            X = baseA(image=X)['image']
            return X, self.y[idx]
        
        else:
            X = A.Normalize(mean=(0.5601, 0.5241, 0.5014), std=(0.2331, 0.2430, 0.2456))(image=X)['image']
            X = AP.transforms.ToTensorV2()(image=X)['image']
            return X, fname
    
    def _preprocess(self, X: str)->Image:
        # X = A.CenterCrop(centsize,centsize)(image=X)['image']
        X = A.Resize(basesize,basesize)(image=X)['image']
        return X

    def _get_xy(self):
        if self.isTrain:
            train_root = os.path.join(self.dataroot, 'train')
            for cls in range(self.n_class):
                img_paths = glob(f'{train_root}/{cls}/**/*.*', recursive=recur)
                self.x.extend( img_paths )
                self.y.extend( [cls] * len(img_paths) )
        
        # Test
        else:
            test_root = os.path.join(self.dataroot, 'test')
            self.x = glob(f'{test_root}/*.*')


class NormalDataset_oversampled(Dataset):
    def __init__(self, isTrain):
        self.x = []
        self.y = []
        self.dataroot = dataroot
        self.n_class = 18
        self.isTrain = isTrain
        self.expand_ratio = 0.5
        self._get_xy()
        self.supl_trans = None

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        fname = self.x[idx].split('/')[-1]
        X = np.array(Image.open(self.x[idx]))
        X = self._preprocess(X)
        
        if self.isTrain:
            X = baseA(image=X)['image']
            return X, self.y[idx]
        
        else:
            X = baseA(image=X)['image']
            return X, fname
    
    def _preprocess(self, X: str)->Image:
        # X = A.CenterCrop(centsize,centsize)(image=X)['image']
        X = A.Resize(basesize,basesize)(image=X)['image']
        return X

    def _get_xy(self):
        if self.isTrain:
            cls_dist = []
            train_root = os.path.join(self.dataroot, 'train')
            for cls in range(self.n_class):
                img_paths = glob(f'{train_root}/{cls}/*.*')
                cls_dist.append(len(img_paths))
                self.x.extend( img_paths )
                self.y.extend( [cls] * len(img_paths) )
            
            supl_nums = self.get_supl_num(cls_dist)
            for cls, supl_num in enumerate(supl_nums):
                img_paths = glob(f'{train_root}/{cls}/*.*')
                supl_paths = random.choices(img_paths, k=supl_num)
                self.x.extend( supl_paths )
                self.y.extend( [cls] * len(supl_paths) )
        
        # Test
        else:
            test_root = os.path.join(self.dataroot, 'test')
            self.x = glob(f'{test_root}/*.*')

    def get_supl_num(self, cls_dist: list)->list:
        cls_dist = np.array(cls_dist)
        supl_ratio = np.log( (cls_dist / np.max(cls_dist)) * 1e5)
        supl_ratio = supl_ratio / np.max(supl_ratio)

        threshold = np.zeros(shape=cls_dist.shape)
        threshold.fill(1500)

        supl_nums = (self.expand_ratio * (np.max(cls_dist) * supl_ratio * (cls_dist < threshold))).astype(np.int)

        return supl_nums



class ProjectedDataset(Dataset):
    def __init__(self, name, isTrain):
        self.x = []
        self.y = []
        self.dataroot = dataroot
        self.name = name
        self.n_class = 18
        self.isTrain = isTrain
        self._get_xy()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        fname = self.x[idx].split('/')[-1]
        X = np.array(Image.open(self.x[idx]))
        X = self._preprocess(X)
        if self.isTrain:
            X = transform_dict[self.name](image=X)['image']
            return X, self.y[idx]
        
        else:
            return X, fname
    
    def _preprocess(self, X)->Image:
        # X = A.CenterCrop(centsize,centsize)(image=X)['image']
        X = A.Resize(basesize,basesize)(image=X)['image']
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
                
                img_paths = glob(f'{train_root}/{cls}/**/*.*', recursive=recur)
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