import os
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from os import listdir
from skimage import io
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset


class MSTAR_10(Dataset):
    def __init__(self, root_dir, train, sar_transform=None):
        """
        Args:
            root_dir (str): 데이터의 루트 디렉토리 (예: "MSTAR_10")
            transform (callable, optional): 데이터 변환(예: ToTensor())
        """
        self.train = train
        self.transform = sar_transform
        sub_dir = "train" if self.train else "test"

        # ImageFolder를 활용하여 데이터 로드
        self.data = datasets.ImageFolder(
            root=os.path.join(root_dir, sub_dir),  # "root_dir/train" 또는 "root_dir/test"
        )
        self.img, self.targets = zip(*self.data.imgs)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 데이터 경로와 라벨 가져오기
        image = Image.open(self.img[idx])
        image = image.convert("L")
        
        if self.transform is not None:
            image = self.transform(image)
            
        label = self.targets[idx]
            
        return {"sar": image, "label": label}

    
class WHU_OPT_SAR(Dataset):
    def __init__(self, root_dir, train=True, optical_transform=None, sar_transform=None):
        """
        Args:
            root_dir (str): 데이터 루트 디렉토리 (예: "data/")
            train (bool): Train 데이터를 로드할지 여부. True면 train, False면 test.
            transform (callable, optional): 데이터 변환 (예: ToTensor())
        """
        self.root_dir = root_dir
        self.train = train
        self.optical_transform = optical_transform
        self.sar_transform = sar_transform

        # Train/Test 디렉토리 선택
        sub_dir = "train" if self.train else "test"
        base_dir = os.path.join(self.root_dir, sub_dir)

        # Optical, SAR, Label 디렉토리 경로 설정
        # root/train/opt, root/train/sar, root/train/lbl or root/test/opt, root/test/sar, root/test/lbl
        
        self.opt_dir = os.path.join(base_dir, "opt")
        self.sar_dir = os.path.join(base_dir, "sar")
        self.lbl_dir = os.path.join(base_dir, "lbl")

        # Optical 데이터를 기준으로 파일 리스트 생성
        self.file_names = sorted(os.listdir(self.opt_dir))  # 모든 파일 이름 (opt 기준)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # 현재 인덱스의 파일 이름
        file_name = self.file_names[idx]

        # Optical, SAR, Label 파일 경로
        opt_path = os.path.join(self.opt_dir, file_name)
        sar_path = os.path.join(self.sar_dir, file_name)
        lbl_path = os.path.join(self.lbl_dir, file_name)

        # 이미지 로드
        optical_image = Image.open(opt_path).convert("RGB")  # Optical은 RGB로 로드
        sar_image = Image.open(sar_path).convert("L")        # SAR은 흑백으로 로드
        #label_image = Image.open(lbl_path).convert("L")      # Label은 원본 그대로 로드
        #optical_image = self.load_tif_image(opt_path, mode="RGB")  # Optical은 RGB로 로드
        #sar_image = self.load_tif_image(sar_path, mode="L")       # SAR은 Grayscale로 로드
        label_image = self.load_label_image(lbl_path)     # Label은 Grayscale로 로드
        
        # Transform 적용
        if self.optical_transform:
            optical_image = self.optical_transform(optical_image)
        if self.sar_transform:
            sar_image = self.sar_transform(sar_image)

        return {"optical": optical_image, "sar": sar_image, "label": label_image}

    def load_label_image(self,path):
        """
        레이블 TIFF 파일 로드 및 정수형 텐서로 변환.
        Args:
            path (str): 레이블 TIFF 파일 경로
        Returns:
            torch.Tensor: 정수형 텐서 ([H, W], dtype=torch.long)
        """
        with Image.open(path) as img:
            label_array = np.array(img, dtype=np.int64)  # NumPy 배열로 변환
            label_tensor = torch.from_numpy(label_array)  # PyTorch 텐서로 변환
            return label_tensor
        
        
class V2Dataset(Dataset):
    def __init__(self, root_dir, train=True, optical_transform=None, sar_transform=None):
        """
        Args:
            root_dir (str): 데이터의 루트 디렉토리 (예: "data/")
            train (bool): True일 경우 train 데이터, False일 경우 test 데이터 로드
            transform (callable, optional): 데이터 변환 (예: ToTensor())
        """
        self.root_dir = root_dir
        self.train = train
        self.optical_transform = optical_transform
        self.sar_transform = sar_transform
        self.data = []

        # Train/Test 디렉토리 설정
        sub_dir = "train" if self.train else "test"
        base_dir = os.path.join(self.root_dir, sub_dir)

        # 클래스 디렉토리 추출
        #self.classes = sorted(os.listdir(base_dir))
        self.classes = [cls for cls in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, cls)) and cls != '.DS_Store']

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Optical(s1)과 SAR(s2) 데이터를 매핑하여 로드
        for cls_name in self.classes:
            class_dir = os.path.join(base_dir, cls_name)
            s1_dir = os.path.join(class_dir, "s1")  # SAR 폴더
            s2_dir = os.path.join(class_dir, "s2")  # Optical 폴더

            # s1과 s2의 파일 이름 리스트 (정렬하여 매칭)
            s1_files = sorted(os.listdir(s1_dir))
            s2_files = sorted(os.listdir(s2_dir))

            for s1_file, s2_file in zip(s1_files, s2_files):
                # (s1 경로, s2 경로, 클래스 라벨) 저장
                self.data.append((
                    os.path.join(s1_dir, s1_file),
                    os.path.join(s2_dir, s2_file),
                    self.class_to_idx[cls_name]
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # s1(Optical), s2(SAR) 경로와 클래스 라벨 가져오기
        s1_path, s2_path, label = self.data[idx]

        # 이미지 로드
        s1_image = Image.open(s1_path).convert("L")  # SAR은 흑백으로 로드
        s2_image = Image.open(s2_path).convert("RGB")  # Optical은 RGB로 로드

        # Transform 적용
        if self.sar_transform:
            s1_image = self.sar_transform(s1_image)
        if self.optical_transform:
            s2_image = self.optical_transform(s2_image)

        return {"sar": s1_image, "optical": s2_image, "label": label}