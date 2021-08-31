import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import re
import os
import argparse

from pathlib import Path
from importlib import import_module
from glob import glob

from dataset_crop import TestDataset
from loss import create_criterion

def seed_everything(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer) :
    for param_group in optimizer.param_groups :
        return param_group['lr']

def grid_image(np_images, gts, preds, n = 16, shuffle = False) :
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k = n) if shuffle else list(range(n))
    figure = plt.figure(figsize = (12, 18 + 2))
    plt.subplots_adjust(top = 0.8)
    n_grid = np.ceil(n ** 5)
    
    for idx, choice in enumerate(choices) :
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]

        title = f'label: {gt}'
        plt.subplot(n_grid, n_grid, idx + 1, title = title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap = plt.cm.binary)
    
    return figure

def increment_path(path, exist_ok = False) :
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()) :
        return str(path)
    else :
        dirs = glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path} {n}"

def train_fold(data_dir, model_dir, args) :
    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # -- dataset
    dataset_module = getattr(import_module('dataset_crop'), args.dataset) # default : MaskBaseDataset
    dataset = dataset_module(
        data_dir = data_dir
    )
    num_classes = dataset.num_classes

    # -- test dataset & loader
    test_dir = '/opt/ml/crop_image/cropped/test'
    submission = pd.read_csv('/opt/ml/input/data/eval/info.csv')
    test_image_paths = [os.path.join(test_dir, img_id) for img_id in submission.ImageID]
    test_dataset = TestDataset(test_image_paths, resize = args.resize, crop = False)
    test_loader = DataLoader(
        test_dataset,
        shuffle = True
    )

    # -- augmentation
    transform_module = getattr(import_module('dataset_crop'), args.augmentation) # default : BaseAugmentation
    if args.augmentation == 'CustomAugmentation_TV' :
        train_transform = transform_module(
                resize = args.resize,
                mean = dataset.mean,
                std = dataset.std,
                train = True
        )
        valid_transform = transform_module(
            resize = args.resize,
            mean = dataset.mean,
            std = dataset.std,
            train = False,
            valid = True
        )
        basic_transform = transform_module(
            resize = args.resize,
            mean = dataset.mean,
            std = dataset.std,
            train = False,
            valid = False
        )
        dataset.set_transform(basic_transform)
    else :
        transform = transform_module(
            resize = args.resize,
            mean = dataset.mean,
            std = dataset.std,
        )
        dataset.set_transform(transform)

    # -- model
    model_module = getattr(import_module('model'), args.model) # default : BaseModel
    model = model_module(
        num_classes = num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion) # default : cross_entropy
    opt_module = getattr(import_module('torch.optim'), args.optimizer) # default : SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = 5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma = 0.5)

    # -- early stopping & out-of-fold option
    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf
    patience = 3
    counter = 0
    oof_pred = None
    results = {}

    # stratified-kfold
    if args.augmentation == 'CustomAugmentation_TV' :
        TV = True
    else :
        TV = False

    if args.fold == 'Stratified_KFold' :
        train_set_list, val_set_list = dataset.stratified_kfold_split_dataset(dataset = dataset, TV = TV, fold = args.fold_number)
    else :
        ValueError("You write another fold method, you need to change method or add it")

    for fold, (train_set, val_set) in enumerate(zip(train_set_list, val_set_list)) :
        print(f'Fold {fold + 1}')
        counter = 0              # 매 fold마다 counter, acc, loss, f1을 초기화하여 다시 Early Stopping 걸리도록 해줌.
        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = np.inf

        # -- dataset & dataloader
        train_set.set_transform(train_transform)
        val_set.set_transform(valid_transform)

        train_loader = DataLoader(
            train_set,
            batch_size = args.batch_size,
            num_workers = 0,
            shuffle = True,
            pin_memory = use_cuda,
            drop_last = True
        )

        val_loader = DataLoader(
            val_set,
            batch_size = args.valid_batch_size,
            num_workers = 0,
            shuffle = False,
            pin_memory = use_cuda,
            drop_last = True
        )

        # -- model
        model = model_module(
            num_classes = num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion) # default : cross_entropy
        opt_module = getattr(import_module('torch.optim'), args.optimizer) # default : SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            weight_decay = 5e-4
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma = 0.5)

        logger = SummaryWriter(log_dir = f"{save_dir}/cv{fold}_model_results")

        for epoch in range(args.epochs) :
            model.train()
            loss_value = 0
            matches = 0
            f1_value = 0

            n_iter = 0
            for idx, train_batch in enumerate(train_loader) :
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                outs = model(inputs)
                preds = torch.argmax(outs, dim = -1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                f1_value += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average = 'macro')
                if (idx + 1) % args.log_interval == 0 : # 20번째에만 출력해라 (log_interval의 default값이 20)
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    train_f1 = f1_value / (args.log_interval * (n_iter + 1))
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 {train_f1:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar('Train/loss', train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar('Train/accuracy', train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

                    n_iter += 1

            scheduler.step()

            # val loop
            with torch.no_grad() :
                print('Calculating validation results ...')
                model.eval()
                val_loss_items = []
                val_acc_items = []
                val_f1_items = []
                figure = None

                n_iter = 0
                for val_batch in val_loader :
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim = -1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    f1_item = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average = 'macro') 

                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    val_f1_items.append(f1_item)

                    if figure is None :
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n = 16, shuffle = args.dataset != 'MaskSplitByProfileDataset'
                        )
                    n_iter += 1

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                val_f1 = np.sum(val_f1_items) / n_iter
                print(f'valdiation loss {val_loss:4.4} || validation accuracy {val_acc:4.2%} || validation f1 score {val_f1:4.2%}')
                
                best_val_loss = min(best_val_loss, val_loss)
                best_val_f1 = max(best_val_f1, val_f1)
                if val_acc > best_val_acc :
                    print(f'New best model for val accuracy : {val_acc:4.2%}! saving the best model..')
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_acc = val_acc
                    counter = 0
                else :
                    counter += 1

                if val_loss < best_val_loss :
                    best_val_loss = val_loss
                    counter = 0

                if counter > patience :
                    print('Early Stopping...')
                    break

                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {float(val_f1):4.2%}|| "
                    f"best acc : {best_val_acc: 4.2%}, best loss: {best_val_loss:4.2}, best f1: {best_val_f1:4.2}"
                )

                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar('Val/accuracy', val_acc, epoch)
                logger.add_figure('result', figure, epoch)
                print()
        
        results[fold] = [best_val_loss, best_val_acc, best_val_f1]

    print(f'{str(args.fold).upper()} CROSS VALIDATION RESULTS FOR {args.fold_number} FOLDS')
    print('------------------------------------')

    average_acc = 0.
    average_loss = 0.
    average_f1 = 0.
    for key, value in results.items() :
        average_acc += value[1]
        average_loss += value[0]
        average_f1 += value[2]
    print(f'Loss Average {average_loss/args.fold_number:4.4} || Acc Average {average_acc/args.fold_number:4.2%} || F1 Average {average_f1/args.fold_number:4.2%}')
    
    #TTA
    all_predictions = []
    with torch.no_grad() :
        for images in test_loader :
            images = images.to(device)

            # TTA
            pred = model(images) / 2               # 원본 이미지 예측
            pred += model(torch.flip(images, [2])) / 2  # flip된 이미지 예측
            all_predictions.extend(pred.cpu().numpy())

        fold_pred = np.array(all_predictions)

    if oof_pred is None :
        oof_pred = fold_pred / args.fold_number
    else :
        oof_pred += fold_pred / args.fold_number

    submission['ans'] = np.argmax(oof_pred, axis = 1)
    submission.to_csv(os.path.join(test_dir, 'submission_3.csv'), index = False)
    print('test inference is done!')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type = int, default = 42, help = 'random seed (default: 42)')
    parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type = str, default = 'MaskBaseDataset', help = 'dataset type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type = str, default = 'BaseAugmentation', help = 'dataset augmentation type (default: BaseAugmentation)')
    parser.add_argument('--resize', nargs='+', type = int, default = [128, 96], help = 'resize size for image when training (default: [128, 96]])')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type = int, default = 1000, help = 'input batch size for validate (default: 1000)')
    parser.add_argument('--model', type = str, default = 'BaseModel', help = 'model type (default: BaseModel)')
    parser.add_argument('--optimizer', type = str, default = 'SGD', help = 'optimizer type (default: SGD)')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type = float, default = 0.2, help = 'ratio for validation (default: 0.2)')
    parser.add_argument('--criterion', type = str, default = 'cross_entropy', help = 'criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type = int, default = 20, help = 'learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type = int, default = 20, help = 'how many batches to wait before logging training (default: 42)')
    parser.add_argument('--name', default = 'exp', help = 'model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type = str, default = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/crop_image/cropped/train'))
    parser.add_argument('--model_dir', type = str, default = os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--fold', type = str, default = 'random_split', help = 'Split Method (default : random_split)')
    parser.add_argument('--fold_number', type = int, default = 4, help = 'Split Fold Number (defualt : 5)')
    args = parser.parse_args()
    print(args) 

    data_dir = args.data_dir
    model_dir = args.model_dir

    if args.fold == 'KFold' or args.fold == 'Stratified_KFold' :
        train_fold(data_dir, model_dir, args)
    #elif args.fold == 'random_split' :
    #    train_basic(data_dir, model_dir, args)
    else :
        ValueError("Fold Method is Wrong, you can use ['random_split', 'kfold', 'stratified-kfold']")



