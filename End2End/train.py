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
n_epochs = 100
lr = 1e-4
loss = 5
f1 = 0.1

model = Networks.Vgg19()
# model = Networks.Efficientnet_b2()
# normal_data = Data.NormalLoader(isTrain=True, batch_size=batch_size)
normal_data = Data.load_data(True, batch_size, name=None, expand=True)
# normal_data = Data.NormalDataset(isTrain=True).get_loader(batch_size=batch_size)
# normal_data = data.ProjectedLoader(name='age', isTrain=True, batch_size=batch_size)

wandb.init(project="vgg19_128", config={
    "learning_rate": lr,
    "architecture": model.__class__,
    "dataset": 'id_sep_val',
})
config = wandb.config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'[{torch.cuda.get_device_name()}]')





# optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
criterion = Losses.FocalLoss(alpha=0.25)
# lr_scheduler = CosineAnnealingLR(optim, T_max=50, eta_min=1e-12)
lr_scheduler = CyclicLR(optim, base_lr=1e-7, step_size_up=5, max_lr=1e-4, 
                                              gamma=0.5, mode='exp_range')

print(f"[{'Epoch':<5} {'LR':<6} {'F1': <5} {'Loss':<6} {'Acc':<5}]")
curr_acc = 0.10
best_accuracy = 0.10
curr_f1 = 0.10
best_f1 = 0.10
for epoch in range(n_epochs):
    model.to(device)
    curr_lr = optim.param_groups[0]['lr']
    for idx, (images, labels) in enumerate(tqdm(normal_data['train'],bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc=f'[{epoch:^5} {curr_lr*1000::^1.4f} {f1*100:>.2f} {loss:^5.4f} {curr_acc*100:3.2f}%]')):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if idx % 26 == 0:
            wandb.log({"learing_rate": lr, 'F1-Score': f1, 'Loss': loss, 'Accuracy': curr_acc})

    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in normal_data['val']:
            images = images.to(device)
            
            outputs = model(images).detach().cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1 = f1_score(labels, predicted, average='macro')
            curr_f1 = f1
        if best_accuracy> correct / total:
            pass

        curr_acc = correct / total
        if best_accuracy < curr_acc or best_f1 < curr_f1:
            best_accuracy = curr_acc
            best_f1 = curr_f1
            if best_accuracy * 100 > 95 and best_f1 *100 > 90:
                Logger.save_weights(state_dict=model.state_dict(), epoch=epoch, f1=f1, acc=best_accuracy)
    wandb.log({'F1-Score': f1, 'Accuracy': curr_acc})
    

    Logger.write_logs(epoch, lr, f1, loss, curr_acc)
    model.train()