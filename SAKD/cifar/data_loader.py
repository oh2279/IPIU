"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

from email.charset import add_alias
import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, DistributedSampler

from sklearn.model_selection import train_test_split

from . import datasets

def calculate_mean_std(loader):
    """
    커스텀 데이터셋의 mean과 std를 계산.

    """

    mean_std_results = {}

    for key in [ "optical", "sar"]:  # 처리할 데이터 키를 지정
        mean = 0
        std = 0
        total_images = 0

        for batch in loader:
            images = batch[key]  # 딕셔너리의 특정 키 값 선택
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_samples

        mean /= total_images
        std /= total_images
        mean_std_results[key] = {"mean": mean.tolist(), "std": std.tolist()}

    print(mean_std_results)

def fetch_dataloader(params,world_size,rank):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
        # Data augmentation
    mean = {
        'V2_opt' : (0.3455, 0.3621, 0.3165),
        'V2_sar' : 0.5200,
        }

    std = {
        'V2_opt' : (0.1746, 0.1672, 0.1570),
        'V2_sar' : 0.1776,
        }
    
    mean_opt = mean['V2_opt']
    std_opt = std['V2_opt']
    mean_sar = mean['V2_sar']
    std_sar = std['V2_sar']

        
    train_optical_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean_opt, std_opt)
    ])

    train_sar_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean_sar, std_sar)
    ])

        
    test_optical_transformer = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_opt, std_opt)
        ])
    
    test_sar_transformer = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_sar, std_sar)
    ])
    
    return test_optical_transformer, test_sar_transformer