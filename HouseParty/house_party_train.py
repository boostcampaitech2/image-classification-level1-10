from EnModels import MaskModel, GenderModel, AgeModel
from options import Options

opt = Options().parse()

n_epoch = 20
lr = 0.001

# mask = MaskModel(opt)
# mask.train(n_epoch, 0.0005)


# # gender = GenderModel(opt)
# # gender.train(n_epoch, 0.005)


age  = AgeModel(opt)
age.train(n_epoch, 0.0004)
