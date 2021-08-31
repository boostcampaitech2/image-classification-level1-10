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

base = '/opt/ml/input/purified/test'
dst =  '/opt/ml/input/cropped/test'
bias = 30

if not os.path.isdir(dst):
    os.makedirs(dst)

W = 384
H = 512
cr_size = 320
cr_w = int((W-cr_size)/2)
cr_h = int((H-cr_size)/2) - 50

tot = len(glob(f'{base}/*.*'))
fault = 0

for im_path in tqdm(glob(f'{base}/*.*')):
    img = Image.open(im_path)

    # img = cv2.imread(im_path)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    boxes,probs = mtcnn.detect(img)
    if isinstance(boxes, np.ndarray):
        boxes = boxes[0]
        _w = int(boxes[0]-bias)
        _h = int(boxes[1]-bias*2)
        w_ = int(boxes[2]+bias)
        h_ = int(boxes[3]+bias/2)
        ww, hh = w_ - _w, h_-_h

        # if not (ww < 162 or 290 < ww) or (hh < 192 or 363 < hh):
        # if (162<=ww and ww<=290) and (192<=hh and hh<=363) and not(_w * _h * w_ * h_ < 0):
        if (155<=ww and ww<=300) and (180<=hh and hh<=370) and not(_w * _h * w_ * h_ < 0):
            # img = img[_h:h_, _w:w_]
            img = img.crop((_w, _h, w_, h_))

        else:
            img = img.crop((cr_w, cr_h, cr_w+cr_size, cr_h+cr_size))
            # img = img[cr_h:cr_h+cr_size , cr_w:cr_w+cr_size]
            fault += 1
    else:
        img = img.crop((cr_w, cr_h, cr_w+cr_size, cr_h+cr_size))
        # img = img[cr_h:cr_h+cr_size , cr_w:cr_w+cr_size]
        fault += 1
    
    tmp = os.path.join(dst, im_path.split('/')[-1])
    img.save(tmp)
    # if img.shape[2] != 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    #     print('changed!!!!!!!!!!!!')
    # plt.imsave(tmp, img)

print(f'{fault / tot * 100}%')

print(len(glob(f'{base}')))
print(len(glob(f'{dst}/**/*.*')))

print(f'{(time.time() - st_time)/60:.2f}min')
