import os
import torch

ex_name = 'Vgg19_128'
log_path = f'./logs/{ex_name}.txt'
print(f'Logging to -> {log_path}')

if not os.path.isdir('./weight'):
    os.makedirs('./weight')
if not os.path.isdir('./logs'):
    os.makedirs('./logs')

def write_logs(epoch, lr, f1, loss, acc):
    if not os.path.isfile(log_path):
        with open(log_path, 'w') as f:
            f.write(f"{'-'*50}\n")
            f.write('Epoch\tLR\t\t\tF1\t\tLoss\t\tAcc\n')
            f.write(f'{epoch+1:<3}\t\t{lr:6f}\t{f1*100:4.2f}\t{loss:.6f}\t{acc*100: .2f}%\n')
    else:
        with open(log_path, 'a') as f:
            f.write(f'{epoch+1:<3}\t\t{lr:6f}\t{f1*100:4.2f}\t{loss:.6f}\t{acc*100: .2f}%\n')


def save_weights(state_dict, epoch, f1, acc):
    ckpt_path = os.path.join('./weight', f'{ex_name}_{epoch+1}_f1-{int(f1*100)}_{int(acc*100)}%.pth')
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict},
        ckpt_path)