import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm
import os

from EnModels import MaskModel, GenderModel, AgeModel
import data
from options import Options

'''
##
# To-do
1. Custom dataset -> 
'''
opt = Options().parse()

### Load Model
Mask = MaskModel(opt)
Gender = GenderModel(opt)
Age = AgeModel(opt)

m_weight = "MASK1_11_97.pth"
g_weight = "MASK1_10_'84.pth"
a_weight = "AgeModel1_6_85.pth"


### Test Model
mas_cls = Mask.test(m_weight)

gen_cls = Gender.test(g_weight, mas_cls)

age_cls = Age.test(a_weight, gen_cls)
