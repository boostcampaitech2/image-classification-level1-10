# pstage_01_image_classification
 
## Dependencies
- `torch==1.7.0`
- `torchvision==0.7.0`                                                              
- `CUDA==11.0`
## Install Requirements
- `pip install -r requirements.txt`
## Hardware
- `GPU : Tesla V100`

## Dataset
**Note :** Dataset의 경우, 보안상의 이유로 공개할 수 없다.
- 이미지 개수 : 전체 4500명에 대하여, 각 7장(마스크 착용5, 이상하게 착용1, 미착용1)의 Image Data 사용. 이 중 60%(18900)을 학습 데이터셋으로 활용한다.
- 이미지 크기 : (384, 512)

데이터 경로는 다음과 같다.
```
datadir (= 'images')
    +- person1 image folder ( = '000001_female_Asian_45')
    |   +- mask1.jpg
    |   +- mask2.jpg
    |   +- mask3.jpg
    |   +- mask4.jpg
    |   +- mask5.jpg
    |   +- incorrect_mask.jpg
    |   +- normal.jpg
    +- person2 image folder
        ...
>> 확장자의 경우 jpg 외에 다양한 확장자가 존재.
```

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
|Argument|type|Defualt|Explanation|
|---|---|---|---|
|--seed|int|42|랜덤에 사용될 시드넘버|
|--epochs|int|5|학습 epoch|
|--dataset|str|MaskDataset|학습에 사용할 데이터셋|
|--lr|float|1e-4|학습의 learning rate|
|--model|str|CustomModel|학습에 사용할 모델|
|--batch_size|int|64|학습의 batch size|
|--valid_batch_size|int|64|검증의 batch size|
|--criterion|str|cross_entropy|학습의 loss function|
|--optimizer|str|SGD|학습의 optimizer|
|--scheduler|str|StepLR|learning rate를 조절하는 스케쥴러|
