import timm
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torchvision.models as models

n_classes = 18

def Efficientnet_b2():
    model = timm.create_model("efficientnet_b2", pretrained=True)
    terminal = model.classifier

    in_feats = terminal.in_features

    terminal = nn.Sequential(# 1408
        nn.Linear(in_features=in_feats, out_features=512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),


        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),

        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),

        nn.Linear(in_features=128, out_features=n_classes)
    )

    for layer in terminal:
        if isinstance(layer, nn.Linear):
            # print(layer)
            nn.init.xavier_uniform_(layer.weight)
            stdv = 1.0 / (layer.in_features ** 0.5)
            layer.bias.data.uniform_(-stdv, stdv)

    for param in model.parameters():
        param.requires_grad = True
    
    return model

# def Efficientnet_b2():
#     model = timm.create_model("efficientnet_b2", pretrained=True)
#     terminal = model.classifier

#     in_feats = terminal.in_features

#     terminal = nn.Sequential(# 1408
#         nn.Linear(in_features=in_feats, out_features=512),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.3, inplace=False),


#         nn.Linear(in_features=512, out_features=512),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.3, inplace=False),

#         nn.Linear(in_features=128, out_features=n_classes)
#     )

#     for layer in terminal:
#         if isinstance(layer, nn.Linear):
#             # print(layer)
#             nn.init.xavier_uniform_(layer.weight)
#             stdv = 1.0 / (layer.in_features ** 0.5)
#             layer.bias.data.uniform_(-stdv, stdv)

#     for param in model.parameters():
#         param.requires_grad = True
    
#     return model


# def Vgg16():
#     model = models.vgg16(pretrained=True)
#     terminal = model.classifier
#     terminal[6].out_features = n_classes
#     # terminal.add_module(
#         # 'terminal',nn.Linear(in_features=terminal[6].out_features, out_features=n_classes))
    
#     stdv = 1 / terminal[6].in_features ** 0.5
#     terminal[6].bias.data.uniform_(-stdv, stdv)
#     for param in model.parameters():
#         param.requires_grad = True

#     return model

def Vgg16():
    model = models.vgg16(pretrained=True)
    terminal = model.classifier
    terminal[6].out_features = n_classes
    # terminal.add_module(
        # 'terminal',nn.Linear(in_features=terminal[6].out_features, out_features=n_classes))
    
    stdv = 1 / terminal[6].in_features ** 0.5
    terminal[6].bias.data.uniform_(-stdv, stdv)
    for param in model.parameters():
        param.requires_grad = True

    return model


def Vgg19():
    model = models.vgg19(pretrained=True)
    terminal = model.classifier
    # terminal[6].out_features = n_classes

    ###################################
    in_feats = terminal[6].in_features
    # terminal[6] = nn.Linear(in_features=in_feats, out_features=n_classes)

    terminal[6] = nn.Sequential(
        nn.Linear(in_features=in_feats, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3, inplace=False),


        nn.Linear(in_features=1024, out_features=256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3, inplace=False),

        nn.Linear(in_features=256, out_features=n_classes)
    )
    # terminal.add_module('7', nn.Softmax(dim=1))
    # terminal.add_module('7', nn.Softmax(dim=0))

    for layer in terminal[6]:
        if isinstance(layer, nn.Linear):
            # print(layer)
            nn.init.xavier_uniform_(layer.weight)
            stdv = 1.0 / (layer.in_features ** 0.5)
            layer.bias.data.uniform_(-stdv, stdv)

    for param in model.parameters():
        param.requires_grad = True

    return model


def Vgg19_revised():
    model = models.vgg19(pretrained=True)
    terminal = model.classifier
    terminal[0] = nn.Linear(in_features=25088, out_features=4096 )
    terminal[3] = nn.Linear(in_features=4096, out_features=1024)
    terminal[6] = nn.Linear(in_features=1024, out_features=256)
    terminal.add_module('7', nn.ReLU(inplace=True))
    terminal.add_module('8', nn.Linear(in_features=256, out_features=256))
    terminal.add_module('9', nn.ReLU(inplace=True))
    terminal.add_module('10', nn.Dropout(p=0.3, inplace=False))
    terminal.add_module('11', nn.Linear(in_features=256, out_features=18))
    
    for layer in terminal:
        if isinstance(layer, nn.Linear):
            print(layer)
            nn.init.xavier_uniform_(layer.weight)
            stdv = 1.0 / (layer.in_features ** 0.5)
            layer.bias.data.uniform_(-stdv, stdv)

    for param in model.parameters():
        param.requires_grad = True

    return model



if __name__ == '__main__':
    model = Efficientnet_b2()
    # model = Vgg19_revised()
    # model = Vgg19()
    print(model)