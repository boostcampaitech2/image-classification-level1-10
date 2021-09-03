import pandas as pd
import numpy as np
import os
import random

from glob import glob
from PIL import Image
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from time import sleep

import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor, Normalize

from torchvision import transforms, utils
from torchvision.transforms import Resize, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler

from dataset import MaskDataset, TestDataset
from model import CustomModel

def seed_everything(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def train(data) :
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is using !')

    LEARNING_RATE = 0.0001
    NUM_EPOCH = 5

    data_path = data['img_path']
    target = np.array(data['label'])
    age_list = np.array(data['age'])
    gender_list = np.array(data['gender'])
    mask_list = np.array(data['stem'])

    from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomBrightnessContrast, GaussNoise, CLAHE, Equalize, ShiftScaleRotate
    from albumentations.pytorch import ToTensorV2
    
    model = CustomModel(18)
    stf = StratifiedKFold(n_splits = 4, shuffle = True, random_state =42)

    for fold, (train_idx, valid_idx) in enumerate(stf.split(data_path, list(target))) :
        print('Fold {}'.format(fold + 1))
        target_array = np.array(target)
        
        dataset_train_Mask = MaskDataset(path_list = data_path[train_idx],
                                        label_list = target[train_idx],
                                        age_list = age_list[train_idx],
                                        gender_list = gender_list[train_idx],
                                        mask_list = mask_list[train_idx],
                                        transform = Compose([
                                                Resize(512, 384, p = 1.0),
                                                Normalize(mean = (0.5,0.5,0.5), std = (0.2, 0.2, 0.2), max_pixel_value = 255.0, p = 1.0),
                                                ToTensorV2(p = 1.0),
                                        ]),
                                        aug_transform = Compose([
                                            Resize(512, 384, p = 1.0),
                                            HorizontalFlip(p = 0.5),
                                            RandomBrightnessContrast(brightness_limit = (-0.3, 0.3), contrast_limit = (-0.3, 0.3), p = 1.0),
                                            GaussNoise(var_limit = (1000, 1600), p = 1.0),
                                            CLAHE(p = 1.0),
                                            Equalize(p = 1.0),
                                            ShiftScaleRotate(p = 1.0),
                                        ]),
                                        need = True
                                        )

        dataset_valid_Mask = MaskDataset(path_list = data_path[valid_idx],
                                        label_list = target[valid_idx],
                                        age_list = age_list[valid_idx],
                                        gender_list = gender_list[valid_idx],
                                        mask_list = mask_list[valid_idx],
                                        transform = Compose([
                                                Resize(512, 384, p = 1.0),
                                                Normalize(mean = (0.5,0.5,0.5), std = (0.2, 0.2, 0.2), max_pixel_value = 255.0, p = 1.0),
                                                ToTensorV2(p = 1.0),
                                        ]))
        BATCH_SIZE = 64
        mask_train_dataloader = torch.utils.data.DataLoader(dataset_train_Mask,
                                                            batch_size = BATCH_SIZE,
                                                            shuffle = True)
        mask_valid_dataloader = torch.utils.data.DataLoader(dataset_valid_Mask,
                                                            batch_size = BATCH_SIZE,
                                                            shuffle = True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

        dataloaders = {
            'train' : mask_train_dataloader,
            'test' : mask_valid_dataloader
        }

        n_epochs_stop = 3
        epochs_no_improve = 0
        early_stop = False
        min_val_loss = np.Inf
        
        best_test_accuracy = 0
        best_test_loss = 9999.

        for epoch in range(5) :
            for phase in ['train', 'test'] :
                running_loss = 0.
                running_acc = 0.
                running_f1 = 0.
                n_iter = 0
                
                if phase == 'train' :
                    model.train()
                elif phase == 'test' :
                    model.eval()

                for ind, (images, labels) in enumerate(tqdm(dataloaders[phase])) :
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train') : # phase == 'train'일 경우에만 grad_enabled를 True
                        logits = model(images)
                        _, preds = torch.max(logits, 1)
                        loss = loss_fn(logits, labels)
                        
                        if phase == 'train' :
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    running_acc += torch.sum(preds == labels.data)
                    running_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average = 'macro')
                    n_iter += 1

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_acc / len(dataloaders[phase].dataset)
                epoch_f1 = running_f1 / n_iter
                
                if phase == 'test' :
                    if epoch_loss < min_val_loss :
                        epochs_no_improve = 0
                        min_val_loss = epoch_loss
                    else :
                        epochs_no_improve += 1
                    
                    if epochs_no_improve == n_epochs_stop :
                        print('Early Stopping!')
                        early_stop = True
                        break

                print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.4f}, 평균 Accuracy : {epoch_acc:.4f}, 평균 F1 Score : {epoch_f1: .4f}")
        
            if early_stop :
                print(f'fold{fold+1} Stopped')
                break

    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    from torchvision import transforms, utils
    from torchvision.transforms import Resize, ToTensor, Normalize

    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        Resize((512, 384), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    model.eval()
    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())

    submission['ans'] = all_predictions
    submission.to_csv(os.path.join(test_dir, 'submission_gogo.csv'), index=False)
    print('test inference is done!')

if __name__ == '__main__' :
    data = pd.read_csv('/opt/ml/final/최종.csv')
    seed_everything(42)
    train(data)
