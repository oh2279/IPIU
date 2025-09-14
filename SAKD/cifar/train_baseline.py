
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
from datetime import datetime
    

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from loss_fun import *
from datasets import WHU_OPT_SAR, V2Dataset
from models.scheduler import CosineAnnealingWarmupRestarts
from tools import seed_all,GradualWarmupScheduler
from spikingjelly.clock_driven import functional,neuron
from sen12ms.classification.dataset import SEN12MS, senToTensor, senNormalize, DatasetTransform
from models import ResNet_cnn,ResNet_snn,ResNet19_cnn,ResNet19_snn, vgg_snn, vgg_cnn, ImageNet_cnn,ImageNet_snn

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', default='None',  help='func_dict')
parser.add_argument('--teacher_name', default='cnn_resnet18',  help='func_dict')
parser.add_argument('--kd', default=False, action='store_true', help='func_dict')
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

def setup(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method='tcp://127.0.0.1:23456')
    torch.cuda.set_device(rank)
    
def cleanup():
    dist.destroy_process_group()


def train_one_epoch(model, teacher, train_loader, loss_fun, optimizer,scheduler):# rank):
        
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    with tqdm(total=len(train_loader), desc=f"Training device {device}") as pbar:
        for iter, batch in enumerate(train_loader):
            imgs, target  = batch
            imgs = imgs.to(device)
            target = target.to(device)
            with autocast():
                if args.kd:
                    with torch.no_grad():
                        out_t, feature_t = teacher(imgs)

                    output, feature_s = model(imgs)

                    ce_loss = loss_fun(output,target)
                    mse_loss = feature_loss(feature_s, feature_t, fun = 'mse')
                    kld_loss = logits_loss(output, out_t, T = 4)

                    loss = ce_loss + mse_loss + kld_loss
                else:
                    output, _  = model(imgs)

                    loss = loss_fun(output,target)


            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # if 'snn' in args.model:
            #     functional.reset_net(model)

            total_loss += loss.item()
            #total_ce_loss = ce_loss.item()
            #total_mse_loss = mse_loss.item()
            #total_kld_loss = kld_loss.item()
            correct += (output.argmax(dim=1) == target).sum().item()
            total_samples += target.size(0)

            # Update tqdm progress
            pbar.set_postfix({"Loss": f"{total_loss / (pbar.n + 1):.4f}", "Accuracy": f"{correct / total_samples:.4f}"})
            pbar.update(1)
    
        scheduler.step()

    #losses = [total_ce_loss/len(train_loader), total_mse_loss/len(train_loader), total_kld_loss/len(train_loader)]
    return total_loss / len(train_loader), correct / total_samples #, losses

def evaluate(model, dataloader, loss_fn):#,rank):

    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    with tqdm(total=len(dataloader), desc=f"Evaluating device {device}") as pbar:
        with torch.no_grad():
            for batch in dataloader:
                imgs, labels = batch
                
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                outputs, _  = model(imgs)
                loss = loss_fn(outputs, labels)

                # Reset the SNN state after each batch
                #if 'snn' in args.model and 'resnet' in args.model:
                    #functional.reset_net(model)

                # Update statistics
                total_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_samples += labels.size(0)

                # Update tqdm progress
                pbar.set_postfix({"Loss": f"{total_loss / (pbar.n + 1):.4f}", "Accuracy": f"{correct / total_samples:.4f}"})
                pbar.update(1)

    return total_loss / len(dataloader), correct / total_samples


#def main_worker(rank, world_size, args):
def main_worker(args):
    best_acc = 0.0
    #setup(rank, world_size)

    model_name = args.model
    model_func = func_dict[model_name]
    model = model_func(num_classes=10).to(device)

    if args.kd:    
        teacher_name = args.teacher_name
        teacher_func = func_dict[teacher_name]
        teacher = teacher_func(num_classes=10).to(device)
        teacher.load_state_dict(torch.load(f'/home/gpuadmin/ipiu2025/SAKD/cifar/cifar10_vgg13/{teacher_name}_best.pth'), strict=False)

    else:
        teacher = None
    
    #model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    trainset = torchvision.datasets.CIFAR10(root='/data/cifar10', train=True, transform=train_transformer, download=True)
    testset = torchvision.datasets.CIFAR10(root='/data/cifar10', train=False, transform=test_transformer, download=True)
   
    #train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    #test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank)
    
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

    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=bathsize,sampler=train_sampler,
    #         shuffle=False, num_workers=4, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(testset, batch_size=bathsize,sampler=test_sampler,
    #         shuffle=False, num_workers=4, pin_memory=True)
    
    learn_rate = lr
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4,nesterov=True)

    #loss_fun = nn.CrossEntropyLoss().cuda(rank)
    loss_fun = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
            
    if args.test:
        print("run test!")
        writer = SummaryWriter(log_dir=f"./simple_avg/runs/cifar10_vgg13/{current_time}")
    else:
        if args.kd: 
            writer = SummaryWriter(log_dir=f"./cifar10_vgg13/kd/{args.model}/{current_time}")
        else:
            writer = SummaryWriter(log_dir=f"./cifar10_vgg13/{args.model}/{current_time}")

    print(f"Training model: {model_name}")
    
    for epoch in range(epochs):
        #train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(model,teacher, train_loader, loss_fun, optimizer, scheduler)#, rank)
        val_loss, val_acc = evaluate(model, val_loader, loss_fun)#, rank)
        
        #if rank ==0:
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch + 1)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch + 1)

        #if rank == 0:  # Log only for rank 0
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Acc: {train_acc}")
        #print(f"Epoch {epoch}, CE Loss: {losses[0]}, MSE Loss: {losses[1]}, KLD Loss: {losses[2]}")
        print(f"Val Loss: {val_loss}, Val Acc: {val_acc}")
        
            
        # 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best accuracy: {best_acc:.4f}")
        print(f"Best Acc: {best_acc}")
    #if rank == 0:
    writer.close()

    #cleanup()
    
if __name__ == '__main__':
    args = parser.parse_args()
    seed = 42
    seed_all(seed)
    
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 형식: "2024-12-09_14-30-00"
    

    if args.test:
        save_dir = f"./simple_avg/cifar10_vgg13/{args.model}"
    elif args.kd:
        save_dir = f"./cifar10_vgg13/kd/{args.model}"
    else:
        save_dir = f"./cifar10_vgg13"
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
        'snn_vgg9': vgg_snn.vgg9,
        'snn_vgg11':vgg_snn.vgg11,
        'snn_vgg13':vgg_snn.vgg13,
        'snn_vgg16':vgg_snn.vgg16,
        'snn_vgg19':vgg_snn.vgg19,
        #'cnn_vgg5':vgg_cnn.vgg5,
        'cnn_vgg9':vgg_cnn.vgg9,
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
    epochs = 200

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = GradScaler()
    #world_size = len(args.gpu.split(","))
    #mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)
    main_worker(args)
    