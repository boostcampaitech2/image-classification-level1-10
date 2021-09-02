import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

from tqdm import tqdm
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

import wandb


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(
        range(batch_size), k=n) if shuffle else list(range(n))
    # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    figure = plt.figure(figsize=(12, 18 + 2))
    # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))
    print("="*50)
    print(f"Model will be saved in {save_dir}")
    print("="*50)
    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()
        print("="*50)
        print("Cuda Cache is empty!!")
        print("="*50)
    # -- dataset
    # default: MaskBaseDataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module(
        "dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module(
        "model"), args.model)  # default: BaseModel

    # Custom transfer model for freezing the backbone parameters
    model = model_module(
        num_classes=num_classes
    ).to(device)
    if type(model).__name__ == "MyModel":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.gender_layer.parameters():
            param.requires_grad = True
        for param in model.age_layer.parameters():
            param.requires_grad = True
        for param in model.mask_layer.parameters():
            param.requires_grad = True
    # model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"),
                         args.optimizer)  # default: SGD
    optimizer_mask = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    optimizer_gender = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    optimizer_age = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )

    scheduler_mask = CosineAnnealingLR(optimizer_mask, T_max=50, eta_min=0)
    scheduler_gender = CosineAnnealingLR(optimizer_gender, T_max=50, eta_min=0)
    scheduler_age = CosineAnnealingLR(optimizer_gender, T_max=50, eta_min=0)
    # scheduler_mask = StepLR(optimizer_mask, args.lr_decay_step, gamma=0.5)
    # scheduler_gender = StepLR(optimizer_gender, args.lr_decay_step, gamma=0.5)
    # scheduler_age = StepLR(optimizer_age, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    config = {"batch_size": args.batch_size,
              "lr": args.lr, "epochs": args.epochs}
    wandb.init(project='Image_Classification_with_distributed',
               entity='woowonjin', config=config)

    best_val_f1 = 0
    best_val_acc = 0
    best_val_mask_loss = np.inf
    best_val_gender_loss = np.inf
    best_val_age_loss = np.inf

    wandb.watch(model)
    for epoch in range(args.epochs):
        # train loop
        model.train()
        # loss_value = 0
        # matches = 0
        mask_loss_value = 0
        gender_loss_value = 0
        age_loss_value = 0
        mask_matches = 0
        gender_matches = 0
        age_matches = 0
        train_f1 = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            # labels = labels.to(device)
            mask_label = labels[0].to(device)
            gender_label = labels[1].to(device)
            age_label = labels[2].to(device)
            labels = (mask_label * 6 + gender_label * 3 + age_label).cpu()

            optimizer_mask.zero_grad()
            optimizer_gender.zero_grad()
            optimizer_age.zero_grad()

            mask_out, gender_out, age_out = model(inputs)
            preds_mask = torch.argmax(mask_out, dim=-1)
            preds_gender = torch.argmax(gender_out, dim=-1)
            preds_age = torch.argmax(age_out, dim=-1)
            preds = (preds_mask * 6 + preds_gender * 3 + preds_age).cpu()
            train_f1 += f1_score(labels, preds, average="macro")

            loss_mask = criterion(mask_out, mask_label, "mask")
            loss_gender = criterion(gender_out, gender_label, "gender")
            loss_age = criterion(age_out, age_label, "age")

            loss_mask.backward()
            loss_gender.backward()
            loss_age.backward()

            optimizer_mask.step()
            optimizer_gender.step()
            optimizer_age.step()

            mask_loss_value += loss_mask.item()
            gender_loss_value += loss_gender.item()
            age_loss_value += loss_age.item()

            mask_matches += (preds_mask == mask_label).sum().item()
            gender_matches += (preds_gender == gender_label).sum().item()
            age_matches += (preds_age == age_label).sum().item()

            if (idx + 1) % args.log_interval == 0:
                train_mask_loss = mask_loss_value / args.log_interval
                train_gender_loss = gender_loss_value / args.log_interval
                train_age_loss = age_loss_value / args.log_interval
                train_mask_acc = mask_matches / args.batch_size / args.log_interval
                train_gender_acc = gender_matches / args.batch_size / args.log_interval
                train_age_acc = age_matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer_mask)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) ||\n"
                    f"training mask loss {train_mask_loss:4.4} || training mask accuracy {train_mask_acc:4.2%} || lr {current_lr}\n"
                    f"training gender loss {train_gender_loss:4.4} || training gender accuracy {train_gender_acc:4.2%} || lr {current_lr}\n"
                    f"training age loss {train_age_loss:4.4} || training age accuracy {train_age_acc:4.2%} || lr {current_lr}\n"
                )
                logger.add_scalar("Train/mask_loss", train_mask_loss,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/gender_loss", train_gender_loss,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/age_loss", train_age_loss,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/mask_accuracy", train_mask_acc,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/gender_accuracy", train_gender_acc,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/age_accuracy", train_age_acc,
                                  epoch * len(train_loader) + idx)

                # loss_value = 0
                # matches = 0
                mask_loss_value = 0
                gender_loss_value = 0
                age_loss_value = 0
                mask_matches = 0
                gender_matches = 0
                age_matches = 0
        print(f"Train f1_score : {train_f1/len(train_loader)}")
        print()
        scheduler_mask.step()
        scheduler_gender.step()
        scheduler_age.step()

        # val loopl
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_mask_loss_items = []
            val_gender_loss_items = []
            val_age_loss_items = []
            val_acc_items = []
            figure = None
            val_f1 = 0
            for val_batch in tqdm(val_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                mask_label = labels[0].to(device)
                gender_label = labels[1].to(device)
                age_label = labels[2].to(device)
                # labels = labels.to(device)
                labels = (mask_label * 6 + gender_label *
                          3 + age_label).to(device)
                mask_out, gender_out, age_out = model(inputs)
                mask_preds = torch.argmax(mask_out, dim=-1)
                gender_preds = torch.argmax(gender_out, dim=-1)
                age_preds = torch.argmax(age_out, dim=-1)
                preds = (mask_preds * 6 + gender_preds *
                         3 + age_preds).to(device)
                mask_loss_item = criterion(mask_out, mask_label, "mask").item()
                gender_loss_item = criterion(
                    gender_out, gender_label, "gender").item()
                age_loss_item = criterion(age_out, age_label, "age").item()

                acc_item = (labels == preds).sum().item()
                val_mask_loss_items.append(mask_loss_item)
                val_gender_loss_items.append(gender_loss_item)
                val_age_loss_items.append(age_loss_item)
                val_acc_items.append(acc_item)
                val_f1 += f1_score(y_true=labels.cpu(),
                                   y_pred=preds.cpu(), average="macro")
                # if figure is None:
                #     inputs_np = torch.clone(inputs).detach(
                #     ).cpu().permute(0, 2, 3, 1).numpy()
                #     inputs_np = dataset_module.denormalize_image(
                #         inputs_np, dataset.mean, dataset.std)
                #     figure = grid_image(
                #         inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                #     )

            val_mask_loss = np.sum(val_mask_loss_items) / len(val_loader)
            val_gender_loss = np.sum(val_gender_loss_items) / len(val_loader)
            val_age_loss = np.sum(val_age_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            val_f1 /= len(val_loader)
            wandb.log({"train_f1": train_f1/len(train_loader),
                      "val_acc": val_acc, "val_f1": val_f1})
            # best_val_mask_loss = min(best_val_mask_loss, val_mask_loss)
            # best_val_gender_loss = min(best_val_gender_loss, val_gender_loss)
            # best_val_age_loss = min(best_val_age_loss, val_age_loss)

            if val_f1 > best_val_f1:
                print()
                print(
                    f"New best model for val f1_score : {val_f1:4.2%}, val acc : {val_acc:4.2%}! saving the best model..")
                print(
                    f"Mask_loss: {val_mask_loss:4.2} , Gender_loss: {val_gender_loss:4.2}, Age_loss: {val_age_loss:4.2} ||")
                torch.save(model.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_mask_loss = val_mask_loss
                best_val_gender_loss = val_gender_loss
                best_val_age_loss = val_age_loss
            else:
                print(
                    f"Model didn't updated!!\n"
                    f"[Val] f1 : {val_f1:4.2%}, acc : {val_acc:4.2%}, Mask_loss: {val_mask_loss:4.2} , Gender_loss: {val_gender_loss:4.2}, Age_loss: {val_age_loss:4.2} ||\n"
                    f"best_f1 : {best_val_f1:4.2%}, best acc : {best_val_acc:4.2%}, best_mask_loss: {best_val_mask_loss:4.2}, best_gender_loss: {best_val_gender_loss:4.2}, best_age_loss: {best_val_age_loss:4.2}"
                )
            torch.save(model.state_dict(), f"{save_dir}/last.pth")
            logger.add_scalar("Val/mask_loss", val_mask_loss, epoch)
            logger.add_scalar("Val/gender_loss", val_gender_loss, epoch)
            logger.add_scalar("Val/age_loss", val_age_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            # logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset',
                        help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list,
                        default=[512, 384], help='resize size for image when training')  # [128, 96], [512, 384]
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64,
                        help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='MyModel',
                        help='model type (default: MyModel)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='custom_focal',
                        help='criterion type (default: custom_focal)')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp',
                        help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get(
        'SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
