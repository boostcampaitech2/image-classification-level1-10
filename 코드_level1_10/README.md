# pstage_01_image_classification
 
## Dependencies
- `torch==1.7.0`
- `torchvision==0.7.0`                                                              
- `CUDA==11.0`
## Install Requirements
- `pip install -r requirements.txt`
## Hardware
- `GPU : Tesla V100`

## Training & Inference
### Structure
- **dataset.py** : Dataset 구조 정의 및 Augmentation 정의
- **loss.py** : Image Classification에 사용될 수 있는 다양한 Loss 정의
- **model.py** : 학습과 추론에 이용될 다양한 모델을 구성
- **train.py** : 학습을 위한 파이프라인 구충 및 검증데이터에 대한 평가 진행

### Implementation
In Terminal
```
python train.py --config {config_path}
# like Underline
# python train.py --optimizer Adam --epochs 10
```


## Arguments Usage
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
