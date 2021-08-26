import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from tqdm import tqdm
import os
import data
import Networks
from options import Options
import pandas as pd

options = Options()
opt = options.parse()

test_cycle = 10

class MaskModel():
    def __init__(self, opt):
        torch.manual_seed(0)
        random.seed(0)
        self.opt = opt
        self.name = 'MaskModel'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.first_HL = 256
        self.model = Networks.SpinalResNet34(self.device, 3).to(self.device)
        self.optimizer = None

    def update_lr(self, lr):    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, n_epoch, lr):
        curr_lr = lr
        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        mask_loader = data.MaskLoader(dataroot=self.opt.dataroot, isTrain=True, batch_size=self.opt.batchsize)
        print('#'*30)
        print(f"\t{mask_loader['train']}\t{len(mask_loader['train'])}")
        print('#'*30)

        for epoch in range(n_epoch):
            for i, (images, labels) in enumerate(tqdm(mask_loader['train'], desc=f'{self.name} ep.{epoch:<3}')):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # print(labels)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                print(f'\t[LOSS]: {loss}')
                correct = 0
                total = 0
                for images, labels in mask_loader['val']:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                if best_accuracy> correct / total:
                    curr_lr = self.opt.learning_rate*np.asscalar(pow(np.random.rand(1),5))
                    self.update_lr(curr_lr)
                    print('Epoch :{} Accuracy SpinalNet: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                                    100 * correct / total, 100*best_accuracy))
                else:
                    best_accuracy = correct / total
                    print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct / total))

                    if best_accuracy * 100 > 70:
                        ckpt_path = os.path.join(self.opt.ckpt_path, self.name, f"{self.name}{self.opt.nMAS}_{epoch+1}_{int(best_accuracy*100)}.pth")
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict()},
                            ckpt_path)
                        print(f'\t{ckpt_path.split("/")[-1]} saved!')

                options.logging(epoch+1, f'{best_accuracy*100: .2f}')

                    
                self.model.train()

    def test(self, weight: str):
        weight_path = os.path.join(opt.ckpt_path, self.name, weight)
        weight = torch.load(weight_path)
        self.model.load_state_dict(weight['state_dict'])
        print(f'{self.name} weight loaded.')

        test_loader = data.MaskLoader(dataroot=self.opt.dataroot, isTrain=False, batch_size=self.opt.batchsize)

        submission = pd.read_csv(self.opt.sub_src)
        self.model.eval()
        with torch.no_grad():
            
            for idx, images, fname in enumerate(tqdm(test_loader, desc=f'{self.name} test')):

                images = images.to(self.device)
                # labels = labels.to(self.device)
        
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                for i in range(predicted.shape[0]):
                    submission[fname[i]] = int(predicted[i]) * 6
                
                if idx > test_cycle:
                    return submission
        # submission.to_csv(f'./submission.csv', index=False)
        # print('test inference is done!')
        return submission



class GenderModel():
    def __init__(self, opt):
        torch.manual_seed(0)
        random.seed(0)
        self.opt = opt
        self.name = 'GenderModel'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.first_HL = 512
        self.model = Networks.SpinalResNet34(self.device, 2, self.first_HL).to(self.device)

    def update_lr(self, lr):    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, n_epoch, lr):
        curr_lr = lr
        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        mask_loader = data.GenderLoader(dataroot=self.opt.dataroot, isTrain=True, batch_size=self.opt.batchsize)
        for epoch in range(n_epoch):
            for i, (images, labels) in enumerate(tqdm(mask_loader['train'], desc=f'Mask epoch {epoch:<3}')):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                print(f'\t[LOSS]: {loss}')
                correct = 0
                total = 0
                for images, labels in mask_loader['val']:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                if best_accuracy> correct / total:
                    curr_lr = self.opt.learning_rate*np.asscalar(pow(np.random.rand(1),5))
                    self.update_lr(curr_lr)
                    print('Epoch :{} Accuracy SpinalNet: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                                    100 * correct / total, 100*best_accuracy))
                else:
                    best_accuracy = correct / total
                    print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct / total))

                    if best_accuracy * 100 > 70:
                        ckpt_path = os.path.join(self.opt.ckpt_path, self.name, f"{self.name}{self.opt.nGEN}_{epoch+1}_{int(best_accuracy*100)}.pth")
                        # ckpt_path = os.path.join(self.opt.ckpt_path, f"{self.opt.model_name + str(self.opt.ex_num) + '_'+ str(epoch+1)+'_'}{int(best_accuracy*100)}.pth")
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict()},
                            ckpt_path)
                        print(f'\t{ckpt_path.split("/")[-1]} saved!')

                options.logging(epoch+1, f'{best_accuracy*100: .2f}')
                    
                # model1.train()
                self.model.train()

    def test(self, weight: str, submission: pd.DataFrame):
        weight_path = os.path.join(opt.ckpt_path, self.name, weight)
        weight = torch.load(weight_path)
        self.model.load_state_dict(weight['state_dict'])
        print(f'{self.name} weight loaded.')

        test_loader = data.GenderLoader(dataroot=self.opt.dataroot, isTrain=False, batch_size=self.opt.batchsize)


        self.model.eval()
        with torch.no_grad():
            
            for idx, images, fname in enumerate(tqdm(test_loader, desc=f'{self.name} test')):

                images = images.to(self.device)
                # labels = labels.to(self.device)
        
                outputs = self.model(images).detach()
                _, predicted = torch.max(outputs.data, 1)

                for i in range(predicted.shape[0]):
                    submission[fname[i]] = int(predicted[i]) * 3 + submission[fname[i]]
                
                if idx > test_cycle:
                    return submission
        # submission.to_csv(f'./submission.csv', index=False)
        # print('test inference is done!')
        return submission



class AgeModel():
    def __init__(self, opt):
        torch.manual_seed(0)
        random.seed(0)
        self.opt = opt
        
        self.name = 'AgeModel'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.first_HL = 512 #256
        self.model = Networks.SpinalResNet34(self.device, 3, 512).to(self.device)

    def update_lr(self, lr):    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, n_epoch, lr):
        curr_lr = lr
        best_accuracy = 0
        criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        mask_loader = data.AgeLoader(dataroot=self.opt.dataroot, isTrain=True, batch_size=self.opt.batchsize)
        for epoch in range(n_epoch):
            for i, (images, labels) in enumerate(tqdm(mask_loader['train'], desc=f'Mask epoch {epoch:<3}')):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                print(f'\t[LOSS]: {loss}')
                correct = 0
                total = 0
                for images, labels in mask_loader['val']:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                if best_accuracy> correct / total:
                    curr_lr = self.opt.learning_rate*np.asscalar(pow(np.random.rand(1),5))
                    self.update_lr(curr_lr)
                    print('Epoch :{} Accuracy SpinalNet: ({:.2f}%), Maximum Accuracy: {:.2f}%'.format(epoch, 
                                                    100 * correct / total, 100*best_accuracy))
                else:
                    best_accuracy = correct / total
                    print('Test Accuracy of SpinalNet: {} % (improvement)'.format(100 * correct / total))

                    if best_accuracy * 100 > 70:
                        ckpt_path = os.path.join(self.opt.ckpt_path, self.name, f"{self.name}{self.opt.nAGE}_{epoch+1}_{int(best_accuracy*100)}.pth")
                        # ckpt_path = os.path.join(self.opt.ckpt_path, f"{self.opt.model_name + str(self.opt.ex_num) + '_'+ str(epoch+1)+'_'}{int(best_accuracy*100)}.pth")
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict()}, 
                            ckpt_path)
                        print(f'\t{ckpt_path.split("/")[-1]} saved!')

                options.logging(epoch+1, f'{best_accuracy*100: .2f}')
                    
                # model1.train()
                self.model.train()

    def test(self, weight: str, submission: pd.DataFrame):
        weight_path = os.path.join(opt.ckpt_path, self.name, weight)
        weight = torch.load(weight_path)
        self.model.load_state_dict(weight['state_dict'])
        print(f'{self.name} weight loaded.')

        test_loader = data.AgeLoader(dataroot=self.opt.dataroot, isTrain=False, batch_size=self.opt.batchsize)


        self.model.eval()
        with torch.no_grad():
            
            for idx, images, fname in enumerate(tqdm(test_loader, desc=f'{self.name} test')):

                images = images.to(self.device)
                # labels = labels.to(self.device)
        
                outputs = self.model(images).detach()
                _, predicted = torch.max(outputs.data, 1)

                for i in range(predicted.shape[0]):
                    submission[fname[i]] = int(predicted[i]) + submission[fname[i]]


                
                if idx > test_cycle:
                    return submission.to_csv(f'./submission.csv', index=False)
        submission.to_csv(f'./submission.csv', index=False)
        print('test inference is done!')
