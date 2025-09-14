
# -*- coding: utf-8 -*-
import os
import math

import torch
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from time import time
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from loss_fun import *
from tools import seed_all
from models import ResNet_cnn,ResNet_snn,ResNet19_cnn,ResNet19_snn, vgg_snn, vgg_cnn, ImageNet_cnn,ImageNet_snn, linear_probing

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', default='snn_101',  help='func_dict')
parser.add_argument('--gpu', default='0',  help='func_dict')
parser.add_argument('--batch_size', default=128,  type=int,help='func_dict')
parser.add_argument('--test', default=False, action='store_true', help='func_dict')

train_transformer = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((32, 32), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

test_transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

# train_transformer = torchvision.transforms.Compose([
#         torchvision.transforms.RandomCrop((32, 32), padding=4),
#         torchvision.transforms.RandomHorizontalFlip(),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
#     ])

# test_transformer = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408),(0.2675, 0.2565, 0.2761))
#     ])

def train_one_epoch(model, lp_models, train_loader, loss_fun, optimizer,scheduler):# rank):
        
    model.eval()
    for lp_model in lp_models:
        lp_model.train()  # Linear probing models will be trained

    total_losses = [0] * len(lp_models)
    total_correct = [0] * len(lp_models)
    total_samples = 0


    with tqdm(total=len(train_loader), desc=f"Training device {device}") as pbar:
        for imgs, target in train_loader:
            imgs, target = imgs.to(device), target.to(device)

            with torch.no_grad():
                _, features = model(imgs)

            # Train each linear probing model
            for i, lp_model in enumerate(lp_models):
                optimizer[i].zero_grad()

                # Forward pass through the linear probing model
                output = lp_model(features[i])

                # Compute loss
                loss = loss_fun(output, target)
                loss.backward()

                # Update weights
                optimizer[i].step()

                total_losses[i] += loss.item()
                preds = output.argmax(dim=1)
                total_correct[i] += (preds == target).sum().item()
                
            total_samples += target.size(0)
            pbar.update(1)


            # Normalize losses by the number of batches
    avg_losses = [total_loss / len(train_loader) for total_loss in total_losses]
    accuracies = [correct / total_samples for correct in total_correct]

    # Update schedulers
    for sched in scheduler:
        sched.step()

    return avg_losses, accuracies

def evaluate(model, lp_models, dataloader, loss_fun):
    model.eval()
    for lp_model in lp_models:
        lp_model.eval()

    total_losses = [0] * len(lp_models)
    total_correct = [0] * len(lp_models)
    total_samples = 0

    with tqdm(total=len(dataloader), desc="Evaluating Linear Probing Models") as pbar:
        with torch.no_grad():
            for imgs, target in dataloader:
                imgs, target = imgs.to(device), target.to(device)

                # Extract features
                _, features = model(imgs)

                for i, lp_model in enumerate(lp_models):
                    output = lp_model(features[i])

                    # Compute loss
                    loss = loss_fun(output, target)
                    total_losses[i] += loss.item()

                    # Compute accuracy
                    total_correct[i] += (output.argmax(dim=1) == target).sum().item()

                total_samples += target.size(0)
                pbar.update(1)

    avg_losses = [total_loss / len(dataloader) for total_loss in total_losses]
    accuracies = [correct / total_samples for correct in total_correct]

    return avg_losses, accuracies


def main_worker(args):

    model_name = args.model
    model_func = func_dict[model_name]
    model = model_func(num_classes=10).to(device)
    
    lp1 = linear_probing.LinearProbe(input_dim=64, num_classes=10).to(device)
    lp2 = linear_probing.LinearProbe(input_dim=128, num_classes=10).to(device)
    lp3 = linear_probing.LinearProbe(input_dim=256, num_classes=10).to(device)
    lp4 = linear_probing.LinearProbe(input_dim=512, num_classes=10).to(device)
    lp_models = [lp1, lp2, lp3, lp4]
    best_accs = [0.0] * len(lp_models)
    #model.load_state_dict(torch.load(f'/home/gpuadmin/ipiu2025/SAKD/cifar/cifar10_model_weight/kd/{model_name}/{model_name}_best.pth'), strict=False)
    #model.load_state_dict(torch.load(f'/home/gpuadmin/ipiu2025/SAKD/cifar/cifar10_model_weight/kd/{model_name}/{model_name}_best1.pth'), strict=False)
    #model.load_state_dict(torch.load(f'/home/gpuadmin/ipiu2025/SAKD/cifar/cifar10_model_weight/kd/{model_name}/{model_name}_best2.pth'), strict=False)
    #model.load_state_dict(torch.load(f'/home/gpuadmin/ipiu2025/SAKD/cifar/cifar10_model_weight/kd/{model_name}_best.pth'), strict=False)
    #model.load_state_dict(torch.load(f'/home/gpuadmin/ipiu2025/SAKD/cifar/cifar10_model_weight/{model_name}/{model_name}_best.pth'), strict=False)
    model.load_state_dict(torch.load(f'/home/gpuadmin/ipiu2025/SAKD/cifar/cifar10_model_weight/{model_name}_best.pth'), strict=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    
    trainset = torchvision.datasets.CIFAR10(root='/data/cifar10', train=True, transform=train_transformer, download=True)
    testset = torchvision.datasets.CIFAR10(root='/data/cifar10', train=False, transform=test_transformer, download=True)
   
    
    train_loader = DataLoader(trainset, 
                                   batch_size=batchsize, #sampler=train_sampler,
                                   num_workers=4, 
                                   #shuffle=False,
                                   shuffle=True, 
                                   pin_memory=True)
    val_loader = DataLoader(testset, 
                                 batch_size=batchsize, #sampler=test_sampler,
                                 num_workers=4, 
                                 shuffle=False, 
                                 pin_memory=True)

    learn_rate = lr
    # Define optimizers and schedulers for each linear probing model
    optimizers = [torch.optim.SGD(params=lp_model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4,nesterov=True) for lp_model in lp_models]
    schedulers = [CosineAnnealingLR(opt, T_max=epochs, eta_min=0) for opt in optimizers]
    loss_fun = nn.CrossEntropyLoss().to(device)

    print(f"Training model: {model_name}")
    
    for epoch in range(epochs):
        #train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(model,lp_models, train_loader, loss_fun, optimizers, schedulers)#, rank)
        val_loss, val_acc = evaluate(model, lp_models,val_loader, loss_fun)#, rank)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, (loss, acc) in enumerate(zip(train_loss, train_acc)):
            print(f"Feature Map {i+1} - Train Loss: {loss:.4f}, Train Acc: {acc:.4f}")
        for i, (loss, acc) in enumerate(zip(val_loss, val_acc)):
            print(f"Feature Map {i+1} - Val Loss: {loss:.4f}, Val Acc: {acc:.4f}")

            # Update best accuracy for the current feature map
            if val_acc[i] > best_accs[i]:
                best_accs[i] = val_acc[i]
                print(f"New Best Accuracy for Feature Map {i+1}: {best_accs[i]:.4f}")

    # Print final best accuracies
    print("\nFinal Best Accuracies for Each Feature Map:")
    for i, best_acc in enumerate(best_accs):
        print(f"Feature Map {i+1} - Best Accuracy: {best_acc:.4f}")
    
if __name__ == '__main__':
    args = parser.parse_args()
    seed = 42
    seed_all(seed)

    func_dict = {
        'snn_resnet18': ResNet_snn.ResNet18,
        'snn_resnet19': ResNet19_snn.resnet19_,
        'snn_resnet34': ResNet_snn.ResNet34,
        'snn_resnet50':ResNet_snn.ResNet50,
        'cnn_resnet18':ResNet_cnn.ResNet18,
        'cnn_resnet19':ResNet19_cnn.resnet19,
        'cnn_resnet34':ResNet_cnn.ResNet34,
        'cnn_resnet50':ResNet_cnn.ResNet50,
        #'snn_vgg5': vgg_snn.vgg5,
        #'snn_vgg9': vgg_snn.vgg9,
        'snn_vgg11':vgg_snn.vgg11,
        'snn_vgg13':vgg_snn.vgg13,
        'snn_vgg16':vgg_snn.vgg16,
        'snn_vgg19':vgg_snn.vgg19,
        #'cnn_vgg5':vgg_cnn.vgg5,
        #'cnn_vgg9':vgg_cnn.vgg9,
        'cnn_vgg11':vgg_cnn.vgg11,
        'cnn_vgg13':vgg_cnn.vgg13,
        'cnn_vgg16':vgg_cnn.vgg16,
        'cnn_vgg19':vgg_cnn.vgg19,
    }


    adamw = False
    warm_up = False
    sta_time = time()
    lr = 0.01
    batchsize = args.batch_size
    epochs = 50

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = GradScaler()
    #world_size = len(args.gpu.split(","))
    #mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)
    main_worker(args)
    