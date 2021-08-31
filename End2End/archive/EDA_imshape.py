from PIL import Image
from glob import glob
import os
from tqdm import tqdm
import numpy as np

base = '/opt/ml/input/cropped'

all_imgs = glob(f'{base}/**/*.*', recursive=True)
# print(len(all_imgs))

# src = ''

min_w = 700
min_h = 700
max_w = 0
max_h = 0
for im_path in tqdm(all_imgs):
    img = Image.open(im_path)
    w, h = img.size

    if w < min_w:
        min_w = w
    if h < min_h:
        min_h = h

    if w > max_w:
        max_w = w
    if h > max_h:
        max_h = h
    
    
# print(min_w, min_h)
print(f'min_w: {min_w}\tmin_h: {min_h}')
print(f'max_w: {max_w}\tmax_h: {max_h}')


##162, 192

# means = []
# stds = []
# for im_path in tqdm(all_imgs):
#     img = np.array(Image.open(im_path))
#     # print(img.shape)
#     # break
#     means.append(img.mean(axis=(0,1)))
#     stds.append(img.std(axis=(0,1)))

# print(np.mean(means, axis=0) / 255.)
# print(np.mean(stds, axis=0) / 255.)

# print(len(all_imgs))