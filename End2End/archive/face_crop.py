import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
from glob import glob
from tqdm import tqdm
from PIL import Image
import time

st_time = time.time()

device = 'cuda'
mtcnn = MTCNN(keep_all=True, device=device)

base = '/opt/ml/input/purified/train'
dst =  '/opt/ml/input/cropped/train'
bias = 30

if not os.path.isdir(dst):
    os.makedirs(dst)
for cls in range(18):
    ndir = os.path.join(dst, str(cls))
    if not os.path.isdir(ndir):
        os.makedirs(ndir)


W = 384
H = 512
cr_size = 300
cr_w = int((W-cr_size)/2)
cr_h = int((H-cr_size)/2) - 50

tot = len(glob(f'{base}/**/*.*'))
fault = 0
VIS = False
idx = 0

for cls in range(18):
    cls_base = os.path.join(base, str(cls))
    cls_imgs = glob(f'{cls_base}/*.*')
    # for idx, img_paths in enumerate(cls_imgs):
    for img_paths in tqdm(cls_imgs, desc=f'{cls: >3}'):
        img = Image.open(img_paths)
        # img = cv2.imread(img_paths) BGRA
        # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        boxes,probs = mtcnn.detect(img)
        if isinstance(boxes, np.ndarray):
            boxes = boxes[0]
            _w = int(boxes[0]-bias)
            _h = int(boxes[1]-bias*2)
            w_ = int(boxes[2]+bias)
            h_ = int(boxes[3]+bias/2)
            ww, hh = w_ - _w, h_-_h
            # if (162<=ww and ww<=290) and (192<=hh and hh<=363) and not(_w * _h * w_ * h_ < 0):
            
            if (155<=ww and ww<=300) and (180<=hh and hh<=370) and not(_w * _h * w_ * h_ < 0):
                # img = img[_h:h_, _w:w_]
                img = img.crop((_w, _h, w_, h_))
        
            else:
                # img = img[cr_h:cr_h+cr_size , cr_w:cr_w+cr_size]
                img = img.crop((cr_w, cr_h, cr_w+cr_size, cr_h+cr_size))
                fault += 1
                if VIS:
                    print(f'[{idx}]', end='\t')
                    print((155<=ww and ww<=300) and (180<=hh and hh<=370), end='\t')
                    print(not(_w * _h * w_ * h_ < 0),end='\t')
                    print('false!', end='\t')
                    print(boxes)
        else:
            img = img.crop((cr_w, cr_h, cr_w+cr_size, cr_h+cr_size))
            # img = img[cr_h:cr_h+cr_size , cr_w:cr_w+cr_size]
            fault += 1
            if VIS:
                print(f'[{idx}]', end='\t')
                print((155<=ww and ww<=300) and (180<=hh and hh<=370), end='\t')
                print(not(_w * _h * w_ * h_ < 0),end='\t')
                print('false!', end='\t')
                print(boxes)
        
        tmp = os.path.join(dst, str(cls), img_paths.split('/')[-1])
        # if img.shape[2] != 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        #     print('changed!!!!!!!!!!!!')
        # plt.imsave(tmp, img)
        img.save(tmp)

print(f'{fault / tot * 100}%')

print(len(glob(f'{dst}/**/*.*')))

print(f'{(time.time() - st_time)/60:.2f}min')