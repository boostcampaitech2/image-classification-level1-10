import torch
import wandb
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR

import Data
import Losses
import Logger
import Networks

'''
multi-label classification
lr_scheduler
'''

batch_size = 128
n_epochs = 50
# lr = 5e-5
lr = 1e-2
loss = 5
f1 = 0.1

model = Networks.Efficientnet_b2()
# model = Networks.Efficientnet_b2()
# normal_data = Data.NormalLoader(isTrain=True, batch_size=batch_size)
normal_data = Data.load_data(True, batch_size, name=None, expand=True)
# normal_data = Data.NormalDataset(isTrain=True).get_loader(batch_size=batch_size)
# normal_data = data.ProjectedLoader(name='age', isTrain=True, batch_size=batch_size)

wandb.init(project="vgg19_256", config={
    "learning_rate": lr,
    "architecture": model.__class__,
    "dataset": 'id_sep_val',
    "project": 'Vgg19_cutmix_customLoss',
    "name": 'proper Focal_loss'
})
config = wandb.config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'[{torch.cuda.get_device_name()}]')





# optim = torch.optim.Adam(model.parameters(), lr=lr)
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# criterion = Losses.FocalLoss(alpha=1.25, gamma=2.2)
# criterion = Losses.FocalLogLoss(n_epochs)
criterion = Losses.SoftmaxProbs()
# criterion = (alpha=0.25)
# lr_scheduler = CosineAnnealingLR(optim, T_max=100, eta_min=1e-13)
lr_scheduler = CosineAnnealingLR(optim, T_max=200, eta_min=1e-12)
# lr_scheduler = CyclicLR(optim, base_lr=1e-7, step_size_up=5, max_lr=1e-4, 
#                                               gamma=0.5, mode='exp_range')

softmax_func = torch.nn.Softmax(dim=1)

print(f"[{'Epoch':<5} {'F1': <5} {'Acc':<5}]")
curr_acc = 0.10
best_accuracy = 0.10
curr_f1 = 0.10
best_f1 = 0.10
curr_lr = lr
best_f1_train = 0.10
for epoch in range(n_epochs):
    model.to(device)

    correct, total = 0, 0
    for idx, (images, label_soft) in enumerate(tqdm(normal_data['train'], desc=f'[{epoch:^5} {f1*100:>.2f} {curr_acc*100:3.2f}%]')):
    # for idx, (images, label_a, label_b, ratio) in enumerate(tqdm(normal_data['train'],bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc=f'[{epoch:^5} {curr_lr*1000::^1.4f} {f1*100:>.2f} {loss:^5.4f} {curr_acc*100:3.2f}%]')):
        images = images.to(device)
        labels = label_soft.to(device)
        # label_a = label_a.to(device)
        # label_b = label_b.to(device)
        # ratio = ratio.to(device)
        
        outputs = model(images)
        # loss = criterion(outputs, label_a, label_b, ratio)
        loss = criterion(outputs, labels)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        curr_lr = optim.param_groups[0]['lr']
        wandb.log({'Loss': loss, 'Learning_rate': curr_lr})


        outputs = outputs.detach().cpu()
        _, predicted = torch.max(outputs.data, 1)
        _, labels_hot = torch.max(labels, 1)
        labels_hot = labels_hot.to('cpu')
        # labels = torch.tensor([label_a[i] if ratio[i] <= 0.5 else label_b[i] for i in range(len(label_a))])
        total += labels.size(0)
        correct += (predicted == labels_hot).sum().item()
        f1_train = f1_score(labels_hot, predicted, average='macro')
        wandb.log({'f1_train': f1_train })
        

        if best_f1_train > f1_train:
            lr_scheduler.step()
        else:
            best_f1_train = f1_train
    acc_train = correct / total
    wandb.log({'acc_train': acc_train })

        # print(ratio)
        # if idx > 10:
        #     break
        # print(f'[outputs.shape]:\n\t {outputs.shape}\t {outputs}')
        # # print(f'label.shape: {labels.shape}\t{labels}')
        # print(f'[label_a.shape]:\n\t {label_a.shape}\t{label_a}')
        # print(f'[label_b.shape]:\n\t {label_b.shape}\t{label_b}')
        # print(f'[ratio.shape]:\n\t   {ratio.shape}\t{ratio}')
        # print('#'*30, end='\n\n\n')
        # exit()
    # wandb.log({'Learning_rate': curr_lr})

    # correct_loss, fault_loss = criterion.step_loss()
    # wandb.log({'correct_loss':correct_loss, 'fault_loss': fault_loss})

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in normal_data['val']:
            images = images.to(device)
            
            outputs = model(images).detach().cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # _, picked_label = torch.max(labels, 1)
            # correct += (predicted == picked_label).sum().item()
            correct += (predicted == labels).sum().item()
            f1 = f1_score(labels, predicted, average='macro')
            # f1 = f1_score(picked_label, predicted, average='macro')
            curr_f1 = f1
        if best_accuracy> correct / total:
            lr_scheduler.step()

        curr_acc = correct / total
        if best_accuracy < curr_acc or best_f1 < curr_f1:
            best_accuracy = curr_acc
            best_f1 = curr_f1
            if best_f1 *100 > 75:
                Logger.save_weights(state_dict=model.state_dict(), epoch=epoch, f1=f1, acc=best_accuracy)
    wandb.log({'F1-Score': curr_f1, 'Accuracy': curr_acc})
    

    Logger.write_logs(epoch, lr, f1, loss, curr_acc)
    model.train()