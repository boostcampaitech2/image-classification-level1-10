# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.7.0
- torchvision==0.7.0                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training & Inference
- `python train.py`

### Arguments Usage
- --seed
  - 랜덤에 사용될 시드
  - type = int
  - default = 42
- --epochs
  - 학습의 epoch
  - type = int
  - default = 5
- --dataset
  - 학습에 사용할 데이터셋
  - type = str
  - default = 'MaskDataset'
- --lr
  - 학습의 learning rate
  - type = float
  - default = 1e-4
- --model
  - 학습에 사용할 모델
  - type = str
  - default = 'CustomModel'
- --batch_size
  - 학습의 batch size
  - type = int
  - default = 64
- --valid_batch_size
  - 검증의 batch size
  - type = int
  - default = 64
- --criterion
  - 학습의 loss function
  - type = str
  - default = 'cross_entropy'
- --optimizer
  - 학습의 optimizer
  - type = str
  - default = 'SGD'
- --scheduler
  - learning rate를 조절하는 스케쥴러
  - type = str
  - default = 'StepLR'
