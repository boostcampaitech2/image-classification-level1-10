from something import Subset_check
from aug_method import AugMethod
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Subset, random_split
from sklearn.model_selection import KFold, StratifiedKFold

from glob import glob
from typing import Tuple
from tqdm import tqdm

import os
import numpy as np

class CustomAugmentation_TV :
    def __init__(self, resize, mean, std, crop = False, train = True, valid = False, **args) :
        self.train_transform = AugMethod.train_augmentation(resize = resize, mean = mean, std = std, crop = crop)
        self.valid_transform = AugMethod.valid_augmentation(resize = resize, mean = mean, std = std, crop = crop)
        self.basic_transform = AugMethod.basic_augmentation(resize = resize, mean = mean, std = std)

        self.train = train
        self.valid = valid

    def __call__(self, image) :
        if self.train :
            return self.train_transform(image)
        elif self.valid :
            return self.valid_transform(image)
        else :
            return self.basic_transform(image)

class MaskCropDataset(Dataset) :
    image_paths = []
    label_list = []
    num_classes = 18

    def __init__(self, data_dir, transform = None, mean = (0.548, 0.504, 0.479), std = (0.237, 0.247, 0.246), val_ratio = 0.2) :
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = transform
        self.setup()

    def setup(self) : # data_dir = '/opt/ml/crop_image/cropped/train'
        profiles = glob(self.data_dir + '/*')
        for profile in profiles :
            folder = os.path.join(self.data_dir, profile)
            
            for file_name in glob(folder + '/*') :
                img_path = os.path.join(self.data_dir, profile, file_name)
                self.image_paths.append(img_path)
                self.label_list.append(int(profile.split('/')[-1]))

    def set_transform(self, transform) :
        self.transform = transform

    def __getitem__(self, idx) :
        assert self.transform is not None, ".set_transform 메소드를 이용하여 transform을 주입해주세요."

        image = self.read_image(idx)
        multi_class_label = self.get_label(idx)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self) :
        return len(self.image_paths)

    def read_image(self, idx) :
        image_path = self.image_paths[idx]
        return Image.open(image_path)

    def get_label(self, idx) :
        return self.label_list[idx]

    @staticmethod
    def denormalize_image(image, mean, std) :
        img_cp = image.copy()
        img_cp = ((img_cp * std) + mean) * 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset] :
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

    def kfold_split_dataset(self, fold = 5) :
        kf = KFold(n_splits = fold, shuffle = True)
        
        train_set_list, val_set_list = [], []
        for train_idx, val_idx in kf.split(self) :
            train_set_list.append(Subset(self, train_idx))
            val_set_list.append(Subset(self, val_idx))
        return train_set_list, val_set_list

    def stratified_kfold_split_dataset(self, dataset, TV = False, fold = 5) :
        labels = self.label_list
        train_set_list, val_set_list = [], []

        skf = StratifiedKFold(n_splits = fold, shuffle = True)
        if TV == False :
            for train_idx, val_idx in skf.split(dataset.image_paths, labels) :
                train_set_list.append(Subset(self, train_idx))
                val_set_list.append(Subset(self, val_idx))
        else :
            for train_idx, val_idx in skf.split(dataset.image_paths, labels) :
                train_set_list.append(Subset_check(self, train_idx))
                val_set_list.append(Subset_check(self, val_idx))
        return train_set_list, val_set_list

class MaskCropDataset_Aug(MaskCropDataset) :
    def __init__(self, data_dir, mean = (0.548, 0.504, 0.479), std = (0.237, ), val_ratio = 0.2) :
        super().__init__(data_dir, mean, std, val_ratio)

        self.X = []
        self.y = []
        self.aug_transform = AugMethod.data_augmentation()

        self.add_aug_image()

    def add_aug_image(self) :
        for path, label in tqdm(zip(self.image_paths, self.label_list)) :
            image = Image.open(path)

            self.X.append(image)
            self.y.append(label)
            # if label == 14 or label == 8 :
            #     for _ in range(15) : # 20 -> 15
            #         self.y.append(label)
            #         self.X.append(transforms.ToPILImage()(self.aug_transform(transforms.ToTensor()(image))))
            # if label == 11 or label == 17 :
            #     for _ in range(12) : # 15 -> 12
            #         self.y.append(label)
            #         self.X.append(transforms.ToPILImage()(self.aug_transform(transforms.ToTensor()(image))))
            # if label == 2 or label == 7 :
            #     for _ in range(4) :
            #         self.y.append(label)
            #         self.X.append(transforms.ToPILImage()(self.aug_transform(transforms.ToTensor()(image))))
            # if label == 5 or label == 12 or label == 6 :
            #     for _ in range(3) :
            #         self.y.append(label)
            #         self.X.append(transforms.ToPILImage()(self.aug_transform(transforms.ToTensor()(image))))
            # if label == 15 or label == 9 or label == 10 or label == 16 :
            #     for _ in range(2) :
            #         self.y.append(label)
            #         self.X.append(transforms.ToPILImage()(self.aug_transform(transforms.ToTensor()(image))))

    def __len__(self) :
        len_dataset = len(self.X)
        return len_dataset

    def __getitem__(self, idx) :
        image = self.X[idx]
        label = self.y[idx]
        image_transform = self.transform(image)
        return image_transform, label

class TestDataset(Dataset) :
    def __init__(self, img_paths, resize, crop, mean = (0.548, 0.504, 0.479), std = (0.237, 0.247, 0.246)) :
        self.img_paths = img_paths
        self.transform = AugMethod.test_augmentation(resize, mean, std, crop)

    def __getitem__(self, idx) :
        image = Image.open(self.img_paths[idx])
        if self.transform :
            image = self.transform(image)
        return image

    def __len__(self) :
        return len(self.img_paths)