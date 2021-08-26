import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageEnhance
import os


'''
RGB Mean: [0.56019358 0.52410121 0.501457  ]
RGB Standard Deviation: [0.23318603 0.24300033 0.24567522]

'''
cx = 17
cy = 41
cw = 367
ch = 471

# def load_dataset(dataroot):
#     transform = transforms.Compose([
#         transforms.RandomRotation((0,15)),
#         transforms.CenterCrop((opt.crsize, opt.crsize)),
#         transforms.Resize((opt.isize, opt.isize)),
#         transforms.RandomAutocontrast(0.5),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ToTensor(),
#     ])
#     dataset = ImageFolder(opt.dataroot)

def train_val_dataset(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

# transform = T.Compose([
#         T.RandomRotation((0,15)),
#         T.Resize((128, 128)),
#         T.RandomAutocontrast(0.3),
#         T.RandomHorizontalFlip(0.5),
#         T.ToTensor(),
#     ])

def MaskLoader(dataroot, isTrain, batch_size):
    if isTrain:
        dataset = train_val_dataset(MaskDataset(dataroot, isTrain))

        dataloader = {x: DataLoader(
            dataset = dataset[x],
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4,
            drop_last = isTrain,
        ) for x in ['train', 'val']}
    
    else:
        dataset = MaskDataset(dataroot, isTrain)

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0,
            drop_last = isTrain,
        )
    return dataloader


def GenderLoader(dataroot, isTrain, batch_size):
    if isTrain:
        dataset = train_val_dataset(GenderDataset(dataroot, isTrain))

        dataloader = {x: DataLoader(
            dataset = dataset[x],
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4,
            drop_last = isTrain,
        ) for x in ['train', 'val']}

    else:
        dataset = GenderDataset(dataroot, isTrain)

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0,
            drop_last = isTrain,
        )
    return dataloader


def AgeLoader(dataroot, isTrain, batch_size):
    if isTrain:
        dataset = train_val_dataset(AgeDataset(dataroot, isTrain))

        dataloader = {x: DataLoader(
            dataset = dataset[x],
            batch_size = batch_size,
            shuffle = True,
            num_workers = 4,
            drop_last = isTrain,
        ) for x in ['train', 'val']}

    else:
        dataset = AgeDataset(dataroot, isTrain)

        dataloader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0,
            drop_last = isTrain,
        )
    return dataloader


class MaskDataset(Dataset):
    # dataroot = '/opt/ml/input/purified'
    def __init__(self, dataroot: str, isTrain: bool, n_class=18):
        self.x = []
        self.y = []
        self.isTrain = isTrain
        self.transform = T.Compose([
                            T.RandomRotation((0,15)),
                            T.Resize((128, 128)),
                            T.RandomAutocontrast(0.3),
                            T.RandomHorizontalFlip(0.5),
                            T.ToTensor(),
                        ])
        self.samples = None

        if isTrain:
            dataroot = os.path.join(dataroot, 'train')
            cls0 = [0,1,2,3,4,5]
            cls1 = [6,7,8,9,10,11]
            cls2 = [12,13,14,15,16,17]
            if (len(cls0 + cls1 + cls2) != n_class) or (len(set(cls0+cls1+cls2)) != n_class):
                raise Exception('[MaskDataset Exeption]: Need mental caring')

            for cls in range(n_class):
                if cls in cls0:     m_cls = 0
                elif cls in cls1:   m_cls = 1
                elif cls in cls2:   m_cls = 2
                else:
                    raise Exception('[Dataset Exeption]: check rootdir')

                cls_paths = glob(f'{dataroot}/{cls}/**/*.*', recursive=True)
                print('#'*30)
                print(len(glob(f'{dataroot}/{cls}/*.*')))
                print(len(glob(f'{dataroot}/{cls}/**/*.*', recursive=True)))
                print('#'*30)
                self.x.extend( cls_paths )
                self.y.extend( [m_cls] * len(cls_paths) )
        
        else:
            print('ASDFASDFASDFASDFASDF!@#$%@%#$R')
            testroot = os.path.join(dataroot, 'test', '0')
            print(testroot)
            self.x = glob(f'{testroot}/' + '*.*')            
            self.samples = self.x[:]
            # print(self.samples[0])
            # exit()



    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        X = Image.open(self.x[idx])
        X = self._preprocess(X)

        if self.isTrain:
            if self.transform:    
                X = self.transform(X)
            return X, self.y[idx]
        else:               
            return T.ToTensor()(X), self.x[idx].split('/')[-1]

    def _preprocess(self, X: str)->Image:
        X = X.crop((cx,cy,cw,ch))
        X = ImageEnhance.Contrast(X).enhance(5)
        X = ImageEnhance.Color(X).enhance(0.8)
        return X



class GenderDataset(Dataset):
    def __init__(self, dataroot: str, isTrain: bool, n_class=18):
        self.x = []
        self.y = []
        self.isTrain = isTrain
        self.transform = T.Compose([
        T.RandomRotation((0,15)),
        T.Resize((128, 128)),
        T.RandomAutocontrast(0.3),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
    ])

        if isTrain:
            dataroot = os.path.join(dataroot, 'train')
            cls0 = [0,1,2, 6,7,8, 12,13,14]
            cls1 = [3,4,5, 9,10,11, 15,16,17]
            if (len(cls0 + cls1) != n_class) or (len(set(cls0+cls1)) != n_class):
                raise Exception('[Dataset Exeption]: Need mental caring')

            for cls in range(n_class):
                if cls in cls0:     m_cls = 0
                elif cls in cls1:   m_cls = 1
                else:
                    raise Exception('[GenderDataset Exeption]: check rootdir')

                cls_paths = glob(f'{dataroot}/{cls}/**/*.*', recursive=True)
                self.x.extend( cls_paths )
                self.y.extend( [m_cls] * len(cls_paths) )
        
        else:
            testroot = os.path.join(dataroot, 'test', '0')
            print(testroot)
            self.x = glob(f'{testroot}/' + '*.*')
            self.samples = self.x[:]


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = Image.open(self.x[idx])
        X = self._preprocess(X)

        if self.isTrain:
            if self.transform:
                X = self.transform(X)
            return X, self.y[idx]
        else:               
            return T.ToTensor()(X), self.x[idx].split('/')[-1]


    def _preprocess(self, X: str)->Image:
        X = X.crop((cx,cy,cw,ch))
        X = ImageEnhance.Contrast(X).enhance(5)
        X = ImageEnhance.Color(X).enhance(0.8)
        return X



class AgeDataset(Dataset):
    def __init__(self, dataroot: str, isTrain: bool, n_class=18):
        self.x = []
        self.y = []
        self.isTrain = isTrain
        self.transform = T.Compose([
        T.RandomRotation((0,15)),
        T.Resize((128, 128)),
        T.RandomAutocontrast(0.3),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
    ])

        if isTrain:
            dataroot = os.path.join(dataroot, 'train')
            cls0 = [0,3,6,9,12,15]
            cls1 = [1,4,7,10,13,16]
            cls2 = [2,5,8,11,14,17]
            if (len(cls0 + cls1 + cls2) != n_class) or (len(set(cls0+cls1+cls2)) != n_class):
                raise Exception('[AgeDataset Exeption]: Need mental caring')

            for cls in range(n_class):
                if cls in cls0:     m_cls = 0
                elif cls in cls1:   m_cls = 1
                elif cls in cls2:   m_cls = 2
                else:
                    raise Exception('[Dataset Exeption]: check rootdir')

                cls_paths = glob(f'{dataroot}/{cls}/**/*.*', recursive=True)
                self.x.extend( cls_paths )
                self.y.extend( [m_cls] * len(cls_paths) )
        
        else:
            testroot = os.path.join(dataroot, 'test', '0')
            print(testroot)
            self.x = glob(f'{testroot}/' + '*.*')
            self.samples = self.x[:]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        X = Image.open(self.x[idx])
        X = self._preprocess(X)

        if self.isTrain:    
            if self.transform:
                X = self.transform(X)
            return X, self.y[idx]
        else:               
            return T.ToTensor()(X), self.x[idx].split('/')[-1]


    def _preprocess(self, X: str)->Image:
        X = X.crop((cx,cy,cw,ch))
        X = ImageEnhance.Contrast(X).enhance(5)
        X = ImageEnhance.Sharpness(X).enhance(5)
        X = ImageEnhance.Brightness(X).enhance(0.9)
        X = ImageEnhance.Color(X).enhance(8)
        return X



if __name__ == '__main__':
    transform = T.Compose([
        T.RandomRotation((0,15)),
        # T.CenterCrop((opt.crsize, opt.crsize)),
        T.Resize((128, 128)),
        T.RandomAutocontrast(0.3),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
    ])

    dataroot = '/opt/ml/input/purified/'
    mask = MaskDataset(dataroot, isTrain=True, transform=transform)
    # print(len(mask))
    # exit()
    # gend = GenderDataset(dataroot, isTrain=True, transform=transform)
    # age  = AgeDataset(dataroot, isTrain=True, transform=transform)
    
    # print(f'Mask[0]:\n\t{next(iter(mask))}\n\n')
    # print(f'Gend[0]:\n\t{next(iter(mask))}\n\n')
    # print(f'Age[0]:\n\t{next(iter(mask))}\n\n')

    # print(len(mask))
    
    mask_loader = MaskLoader(dataroot, True, 1)
    print(len(mask_loader['train']) + len(mask_loader['val']))

    gender_loader = GenderLoader(dataroot, True, 1)
    print(len(gender_loader['train']) + len(gender_loader['val']))


    age_loader = AgeLoader(dataroot, True, 1)
    print(len(age_loader['train']) + len(age_loader['val']))