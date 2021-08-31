import os
import timm
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import Data
import Networks

batch_size = 128
weight_base = './weight'
weight_fname = 'oversam_12_f1-98_99%.pth'
weight_path = os.path.join(weight_base, weight_fname)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'[{torch.cuda.get_device_name()}]')

# model = Networks.Efficientnet_b2()

model = timm.create_model("efficientnet_b2", pretrained=False)
terminal = model.classifier
terminal.out_features=18
weight = torch.load(weight_path)
# print(weight['state_dict'])
# exit()
model.load_state_dict(weight['state_dict'])
print(f'[Weight]:\t {weight_path} loaded')

test_data = Data.load_data(isTrain=False, batch_size=batch_size)

pred_dict = {}
model.eval()
model.to(device)
with torch.no_grad():
    for images, fnames in tqdm(test_data):
        images = images.to(device)
        
        outputs = model(images).detach().cpu()
        _, predicted = torch.max(outputs.data, 1)

        for fname, pred in zip(fnames, predicted):
            pred_dict[fname] = int(pred)


sheet = pd.read_csv('/opt/ml/input/data/eval/info.csv')
tmparr = []
for k in sheet['ImageID']:
    tmparr.append(pred_dict[k])

sheet['ans'] = tmparr
sheet.to_csv('./submissions/sub_over_99__e.csv', index=False)
