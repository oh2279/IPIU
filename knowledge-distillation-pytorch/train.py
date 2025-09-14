"""Main entrance for train/eval with/without KD on CIFAR-10"""

import os
import warnings

import time
import math
import random
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from torch.utils.tensorboard import SummaryWriter

import utils
import model.net as net
import model.data_loader as data_loader
import model.resnet as resnet
import model.wrn as wrn
import model.densenet as densenet
import model.resnext as resnext
import model.preresnet as preresnet
import model.sew_resnet as sewresent
from model.spiking_vgg import vgg11,vgg13,vgg16,vgg19
from evaluate import evaluate, evaluate_kd

parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/sar/sewresnet50_student',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'

parser.add_argument('--model_version', default='experiments/sar/sewresnet50_student')
parser.add_argument('--subset_percent', default=1.0)
parser.add_argument('--augmentation', default='yes')
parser.add_argument('--teacher', default=None)
parser.add_argument('--alpha', default=0.5,
                    help="ema between ce and kld")
parser.add_argument('--temperature', default=1)
parser.add_argument('--learning_rate', default=1e-3)
parser.add_argument('--save_summary_steps', default=100)
parser.add_argument('--num_epochs', default=200)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--num_workers', default=4)
parser.add_argument('--dataset', default="V2")

parser.add_argument('--num_classes', default=4)
parser.add_argument('--cuda', default="True")
parser.add_argument('--parallel', default="True")


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):
            # move to GPU if available
            train_opt_batch = batch["optical"].to(device)
            train_sar_batch = batch["sar"].to(device)
            labels_batch = batch["label"].to(device)
            
            #combined_input = torch.cat((train_opt_batch, train_sar_batch), dim=1)  # [B, 4, H, W]
            #output_batch = model(combined_input)
            
            output_batch,_ = model(train_sar_batch)
            
            loss = loss_fn(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
            
    tqdm.write("Epoch completed: Loss = {:.4f}".format(loss_avg()))
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                    loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    # # reload weights from restore_file if specified
    # if restore_file is not None:
    #     restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    #     logging.info("Restoring parameters from {}".format(restore_path))
    #     utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

     #Tensorboard logger setup
    writer = SummaryWriter(f'{args.model_dir}/logs/')
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    for epoch in range(params.num_epochs):
    
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        
        metrics_string = train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        scheduler.step()
        
        writer.add_scalar("Loss/train", metrics_string['loss'],epoch)
        writer.add_scalar("accruacy/train", metrics_string['accuracy'],epoch)
        
        scheduler.step()
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, device)        
        tqdm.write("Epoch {}: Validation Accuracy = {:.4f}".format(epoch + 1, val_metrics['accuracy']))
        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        
        for name, param in model.named_parameters():

            if param.grad is None:
                print(f"Parameter {name} did not receive gradient.")

    writer.close()


    
# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params,TemporalProjections):

    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """
    
    # set model to training mode
    model.train().to(device)
    teacher_model.eval().to(device)
    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    criterion = nn.MSELoss()
    
    def get_layer(model, name):
        layer = model
        for attr in name.split("."):
            layer = getattr(layer, attr)
        return layer

    def set_layer(model, name, layer):
        try:
            attrs, name = name.rsplit(".", 1)
            model = get_layer(model, attrs)
        except ValueError:
            pass
        setattr(model, name, layer)

    # Conv1 가중치 수정 함수
    def adjust_conv1_weights_for_single_channel(model):
        with torch.no_grad():
            # Get the original conv1 layer
            conv1 = get_layer(model, "conv1")
            if not isinstance(conv1, nn.Conv2d):
                raise ValueError("Expected a Conv2d layer for 'conv1'")
            
            # Compute the new weights by averaging across input channels
            new_weights = conv1.weight.data.mean(dim=1, keepdim=True)  # Shape: (out_channels, 1, H, W)
            
            # Create a new Conv2d layer with single input channel
            new_conv1 = nn.Conv2d(
                in_channels=1,  # Adjusted to accept single-channel input
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False
            )
            # Assign the averaged weights to the new Conv2d layer
            new_conv1.weight.data = new_weights
            
            # Replace the old conv1 layer with the new one
            set_layer(model, "conv1", new_conv1)
            print(f"Updated conv1 layer: {new_conv1}")
        
    adjust_conv1_weights_for_single_channel(teacher_model)
    
    TemporalProjections = nn.ModuleList([sewresent.TemporalProjection() for _ in range(4)]).to(device)

    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):

            # move to GPU if available
            if params.dataset == "MSTAR_10":
                train_sar_batch = batch["sar"].to(device)
                labels_batch = batch["label"].to(device)
            else:
                train_opt_batch = batch["optical"].to(device)
                train_sar_batch = batch["sar"].to(device)
                labels_batch = batch["label"].to(device)
            
            # compute model output, fetch teacher output, and compute KD loss
            #train_sar_batch = train_sar_batch.repeat(1, 4, 1, 1)
            output_batch,feature_maps = model(train_sar_batch)

            with torch.no_grad():
                #train_sar_batch = train_sar_batch.repeat(1, 4, 1, 1)
                output_teacher_batch, teacher_feature_maps = teacher_model(train_sar_batch)

            ce_and_kd_loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
            
            feature_distill_loss = 0.0
            

            # Feature Distillation Loss
            for TemporalProjection,student_feature, teacher_feature in zip(TemporalProjections,feature_maps, teacher_feature_maps):
                feature_distill_loss += criterion(TemporalProjection(student_feature), teacher_feature)
                
            loss =  ce_and_kd_loss + params.beta * feature_distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
        
    # compute mean of all metrics in summary
    tqdm.write("Epoch completed: Loss = {:.4f}".format(loss_avg()))
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean

def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, metrics, params, model_dir, restore_file=None):
    
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    #Tensorboard logger setup
    writer = SummaryWriter(f'{args.model_dir}/logs/')
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    TemporalProjections = nn.ModuleList(
        #[DDP(sewresent.TemporalProjection(channels).to(params.rank), device_ids=[params.rank],find_unused_parameters=True) for channels in [256,512,1024,2048]])
        [sewresent.TemporalProjection(channels).to(params.rank) for channels in [256,512,1024,2048]])
    
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        metrics_string = train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader, metrics, params, TemporalProjections)
        scheduler.step()
        
        writer.add_scalar("Loss/train", metrics_string['loss'],epoch)
        writer.add_scalar("accruacy/train", metrics_string['accuracy'],epoch)
        
        # Evaluate for one epoch on validation set
        with torch.no_grad():
            val_metrics = evaluate_kd(model, val_dataloader, metrics, params,device)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc
        
        writer.add_scalar("Loss/test", val_metrics['test_loss'],epoch)
        writer.add_scalar("accruacy/test", val_acc,epoch)
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        #print(model.named_parameters())
        for name, param in model.named_parameters():
            #print(name,param)
            if param.grad is None:
                print(f"Parameter {name} did not receive gradient.")
        #(2) Log values and gradients of the parameters (histogram)
        # for tag, value in model.named_parameters():
        #    tag = tag.replace('.', '/')
        #    board_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #    board_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
    writer.close()

if __name__ == '__main__':

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # 입력 크기가 고정되지 않은 경우 False로 설정
    torch.backends.cudnn.deterministic = True  # 반복 실행에서 동일한 결과를 얻기 위해 설정
    
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    if args.cuda: torch.cuda.manual_seed(42)
    
    gpu_id=args.gpu
    #device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3' # 네 개 device를 사용

    #print("Use GPU: {} for training".format(gpu_id))
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")


    train_dl, dev_dl = data_loader.fetch_dataloader(args,args.world_size,args.rank)

    logging.info("- done.")
    logging.info(f"Number of training samples: {len(train_dl.dataset)}")
    logging.info(f"Number of validation samples: {len(dev_dl.dataset)}")
    """Based on the model_version, determine model/optimizer and KD training mode
       WideResNet and DenseNet were trained on multi-GPU; need to specify a dummy
       nn.DataParallel module to correctly load the model parameters
    """
    if "student" in args.model_version:
        
        if args.model_version == 'sewresnet50_student':
            model = sewresent.sew_resnet50().to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = resnet.metrics
            loss_fkd = nn.MSELoss()
            
        elif args.model_version == 'resnet50_student':
            model = resnet.resnet50()
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = resnet.metrics
            loss_fkd = nn.MSELoss()

        """ 
            Specify the pre-trained teacher models for knowledge distillation
            Important note: wrn/densenet/resnext/preresnet were pre-trained models using multi-GPU,
            therefore need to call "nn.DaraParallel" to correctly load the model weights
            Trying to run on CPU will then trigger errors (too time-consuming anyway)!
        """
        if args.teacher == "resnet18":
            teacher_model = resnet.ResNet18()
            teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
            #teacher_model = nn.DataParallel(teacher_model).cuda()
            
        if args.teacher == "resnet50":
            teacher_model = resnet.resnet50(num_classes=args.num_classes)
            teacher_checkpoint = 'experiments/sar/resnet50_teacher/best.pth.tar'
            #teacher_model = nn.DataParallel(teacher_model).cuda()

        elif args.teacher == "wrn":
            teacher_model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=10,
                                           dropRate=0.3)
            teacher_checkpoint = 'experiments/base_wrn/best.pth.tar'
            #teacher_model = nn.DataParallel(teacher_model).cuda()

        elif args.teacher == "densenet":
            teacher_model = densenet.DenseNet(depth=100, growthRate=12)
            teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
            #teacher_model = nn.DataParallel(teacher_model).cuda()

        elif args.teacher == "resnext29":
            teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=10)
            teacher_checkpoint = 'experiments/base_resnext29/best.pth.tar'
            #teacher_model = nn.DataParallel(teacher_model).cuda()

        elif args.teacher == "preresnet110":
            teacher_model = preresnet.PreResNet(depth=110, num_classes=10)
            teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
            #teacher_model = nn.DataParallel(teacher_model).cuda()

        utils.load_checkpoint(teacher_checkpoint, teacher_model)

        # Train the model with KD
        logging.info("Experiment - model version: {}".format(args.model_version))
        logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
        logging.info("First, loading the teacher model and computing its outputs...")
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                              metrics, args, args.model_dir, args.restore_file)

    # non-KD mode: regular training of the baseline CNN or ResNet-18
    else:
        if args.model_version == "resnet18":
            model = resnet.ResNet18()
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
        
        elif args.model_version == "resnet50":
            model = resnet.resnet50(num_classes=args.num_classes)
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
            
        elif args.model_version == "sewresnet50_teacher":
            model = sewresent.sew_resnet50(num_classes=args.num_classes)
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
            
        elif args.model_version == "sewresnet101":
            model = sewresent.sew_resnet101(num_classes=args.num_classes)
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
        if args.parallel:
            model = nn.DataParallel(model)
        model=model.to(device)
        train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, args,
                           args.model_dir, args.restore_file)
