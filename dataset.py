import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.png',
    '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

def is_image_file(filename) :
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class BaseAugmentation :
    def __init__(self, resize, mean, std, **args) :
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean = mean, std = std)
        ])

    def __call__(self, image) :
        return self.transform(image)

class AddGaussianNoise(object) :
    '''
        transform에 없는 기능들은 이런 식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있다.
    '''
    def __init__(self, mean = 0, std = 1.) :
        self.std = std
        self.mean = mean

    def __call__(self, tensor) :
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) :
        return self.__class__.__name__ + '(mean={0}, std={1}'.format(self.mean, self.std)

class CustomAugmentation:
    def __init__(self, resize, mean, std, **args) :
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean = mean, std = std),
            AddGaussianNoise()
        ])

    def __call__(self, image) :
        return self.transform(image)

class MaskLabels(int, Enum) :
    Mask = 0
    INCORRECT = 0
    NORMAL = 0

class GenderLabels(int, Enum) :
    MALE = 0
    FEMALE = 1

    @classmethod # 정적메소드, class의 내부에서 def를 사용할 때 'cls'인자를 추가하여 class내의 변수를 활용할 수 있다.
    def from_str(cls, value: str) -> int :
        value = value.lower()
        if value == 'male' :
            return cls.MALE # 남자가 들어올 경우 0을 return
        elif value == 'female' :
            return cls.FEMALE # 여자가 들어올 경우 1을 return
        else :
            return ValueError(f"Gender values should be either 'male' or 'female', Now : input string is {value}")

class AgeLabels(int, Enum) :
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int :
        try :
            value = int(value)
        except Exception :
            raise ValueError(f'Age value should be numeric, Now: Value and type are {value} and {type(value)}')

        if value < 30 :
            return cls.YOUNG
        elif value < 60 :
            return cls.MIDDLE
        else :
            return cls.OLD

class MaskBaseDataset(Dataset) :
    num_classes = 3 * 2 * 3

    _file_name = {
        'mask1' : MaskLabels.Mask,
        'mask2' : MaskLabels.Mask,
        'mask3' : MaskLabels.Mask,
        'mask4' : MaskLabels.Mask,
        'mask5' : MaskLabels.Mask,
        'incorrect_mask' : MaskLabels.INCORRECT,
        'normal' : MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean = (0.548, 0.504, 0.479), std = (0.237, 0.247, 0.246), val_ratio = 0.2) :
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self) :
        '''
        폴더 순서대로 각 이미지들의 image_path, mask, age, gender label을 추출한다.
        '''
        profiles = os.listdir(self.data_dir)
        for profile in profiles :
            if profile.startswith('.') : # '.'으로 시작하는 파일을 무시한다.
                continue                # os.listdir로 불러오게 되면 hidden file이 보이기 때문.
        
            img_folder = os.path.join(self.data_dir, profile) # 7장의 이미지가 들어있는 폴더
            for file_name in os.listdir(img_folder) :
                _file_name, ext = os.path.splitext(file_name) # os.path.splitext를 이용하여 확장자만 분리한다.
                if _file_name not in self._file_name : # 우리가 정의해준 file이름에 속하지 않을 경우 무시한다.
                    continue                            # '.'으로 시작하는 파일을 무시한다.

                img_path = os.path.join(self.data_dir, profile, file_name)
                mask_label = self._file_name[_file_name] # dictionary를 활용하여 파일이름을 기준으로 label을 추출한다.

                id, gender, race, age = profile.split('_') # profile(image 7이 들어있는 폴더 이름)
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)
                
                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
    
    def calc_statistics(self) :
        '''
        전체 이미지의 standard deviation과 mean을 구해준다.
        '''
        has_statistics = self.mean is not None and self.std is not None # mean과 std의 값이 있을 경우 True
        if not has_statistics : # mean & std 모두 가질 경우
            print('[Warning] Calculating statistics... It can take a long time depending on your CPU machine')
            sums = []
            squred = []
            for image_path in self.image_paths[:3000] :
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis = (0,1)))          # 이미지 픽셀의 채널별 평균
                squred.append((image ** 2).mean(axis = (0,1))) # 이미지 픽셀값의 제곱의 채널별 평균

            self.mean = np.mean(sums, axis = 0) / 255                            # 전체 이미지의 픽셀 채널별 평균
            self.std = (np.mean(squred, axis = 0) - self.mean ** 2) ** 0.5 / 255 # 전체 이미지의 픽셀 채널별 std
    
    def set_transform(self, transform) :
        self.transform = transform
    
    def __getitem__(self, idx) :
        assert self.transform is not None, ".set_transform 메소드를 이용하여 transform을 주입해주세요."
        # transform가 값을 가지지 않을 경우 False를 반환하고 AssertionError를 출력한다.
        # Error를 출력할 때의 메세지는 위와 같이 지정한다.

        image = self.read_image(idx)
        mask_label = self.get_mask_label(idx)
        gender_label = self.get_gender_label(idx)
        age_label = self.get_age_label(idx)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self) :
        return len(self.image_paths)

    def read_image(self, idx) :
        image_path = self.image_paths[idx]
        return Image.open(image_path)

    def get_mask_label(self, idx) :
        return self.mask_labels[idx]

    def get_gender_label(self, idx) :
        return self.gender_labels[idx]
    
    def get_age_label(self, idx) :
        return self.age_labels[idx]

    @staticmethod # class에 속해있으면서 self를 인자로 받지 않는다.
    def encode_multi_class(mask_label, gender_label, age_label) -> int :
        return mask_label * 6 + gender_label * 3 + age_label
    
    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels] :
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std) :
        img_cp = image.copy()
        img_cp = ((img_cp * std) + mean) * 255.0         # normalize해준 이미지를 원본으로 복구해준다.
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)  # 0보다 작은 값은 0으로, 255보다 큰 값은 255로 만들어주고 타입을 uint8 변경한다.
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset] :
        '''
        데이터셋을 train, val로 나눈다.
        PyTorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눈다.
        '''
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val]) # random_split는 한번 더 이해하기
        return train_set, val_set

    def kfold_split_dataset(self, fold = 5) -> Tuple[list, list] :
        from sklearn.model_selection import KFold
        kf = KFold(n_splits = fold, shuffle = True) # seed는 train에서 맨 처음 정의하고 넘어간다.
        train_set_list, val_set_list = [], []
        for train_idx, val_idx in kf.split(self) :
            train_set_list.append(Subset(self, train_idx))
            val_set_list.append(Subset(self, val_idx))
        return train_set_list, val_set_list 
    
    def Stratified_kfold_split_dataset(self, dataset, fold = 5) -> Tuple[list, list] :
        from sklearn.model_selection import StratifiedKFold
        labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]
        skf = StratifiedKFold(n_splits = fold, shuffle = True) # seed는 train에서 맨 처음 정의하고 넘어간다.
        train_set_list, val_set_list = [], []
        for train_idx, val_idx in skf.split(dataset.image_paths, labels) :
            train_set_list.append(Subset(self, train_idx))
            val_set_list.append(Subset(self, val_idx))
        return train_set_list, val_set_list 

# class MaskBaseDataset_Aug(MaskBaseDataset) :
#     def __init__(self, data_dir, mean = (0.548, 0.504, 0.479), std = (0.237, )) :
#         super().__init__(data_dir, mean, std, val_ratio)
class MaskSplitByProfileDataset(MaskBaseDataset) :
    '''
    train, val을 나누는 기준을 이미지에 대해서 random이 아닌 사람(profile)을 기준으로 나눈다.
    구현은 val_ratio에 맞게 train / val로 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing한다.
    이후 `split_dataset`에서 index에 맞게 Subset으로 dataset을 분기한다.
    '''

    def __init__(self, data_dir, mean = (0.548, 0.504, 0.479), std = (0.237, 0.247, 0.246), val_ratio = 0.2) :
        self.indices = defaultdict(list)
        # key값이 없을 경우 미리 지정해놓은 초기값을 반환하는 dictionary이다.
        # 없는 key값이 들어왔을 경우 빈 list로 반환하라.
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio) :
        '''
        폴더를 기준으로 train, val의 index 번호를 추출한다.
        '''
        length = len(profiles) # 폴더의 개수
        n_val = int(length * val_ratio) # 폴더의 개수를 기준으로 n_val을 구한다.

        val_indices = set(random.choices(range(length), k = n_val)) # weight를 주지 않고 length를 기준으로 k개를 선택한다.
        train_indices = set(range(length)) - val_indices
        return {
            'train' : train_indices,
            'val' : val_indices
        }
    
    def setup(self) :
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith('.')]
        # 이미지 설명이 붙어있는 폴더이름들을 리스트로 가져온다.

        split_profiles = self._split_profile(profiles, self.val_ratio) # dictionary

        cnt = 0
        for phase, indices in split_profiles.items() : # phase(train/val), indices(index number)
            for _idx in indices :
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder) :
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names :
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split('_')
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)
                    
                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt) # {'train' : []} -> train list를 생성하며 거기에 cnt를 추가한다.
                    cnt += 1                        # {'train' : [], 'val' : []} -> val list를 생성해주며 거기에 cnt를 추가한다.

    def split_dataset(self) -> List[Subset] :
        '''
        순서
        random.choice -> make_label/make_image_path(여기서부터 뒤죽박죽으로 만들어짐)
        -> train/val의 list에 순서대로 cnt append -> train/val Subset을 그냥 순서대로 만듬(근데 이미 random.choice가 이루어진 상태)
        '''
        return [Subset(self, indices) for phase, indices in self.indices.items()]

class TestDataset(Dataset) :
    def __init__(self, img_paths, resize, mean = (0.548, 0.504, 0.479), std = (0.237, 0.247, 0.246)) :
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean = mean, std = std)
        ])

    def __getitem__(self, idx) :
        image = Image.open(self.img_paths[idx])
        if self.transform :
            image = self.transform(image)
        return image
    
    def __len__(self) :
        return len(self.img_paths)

# from importlib import import_module
# from pathlib import Path
# dataset_module = getattr(import_module('dataset'), 'MaskBaseDataset') # default : MaskBaseDataset
# dataset = dataset_module(
#     data_dir = '/opt/ml/input/data/train/images'
# )
# #print(list(dataset[1])[1])
# print(dataset[:])