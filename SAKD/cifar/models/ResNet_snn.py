'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = SeqToANNContainer(nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.bn1 = SeqToANNContainer(nn.BatchNorm2d(planes))
        self.conv2 = SeqToANNContainer(nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False))
        self.bn2 = SeqToANNContainer(nn.BatchNorm2d(planes))
        self.act = LIFSpike()
        self.shortcut = nn.Sequential(SeqToANNContainer())
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SeqToANNContainer(nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)),
                SeqToANNContainer(nn.BatchNorm2d(self.expansion*planes))
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = SeqToANNContainer(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn1 = SeqToANNContainer(nn.BatchNorm2d(planes))
        self.conv2 = SeqToANNContainer(nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False))
        self.bn2 = SeqToANNContainer(nn.BatchNorm2d(planes))
        self.conv3 = SeqToANNContainer(nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False))
        self.bn3 = SeqToANNContainer(nn.BatchNorm2d(self.expansion*planes))
        self.act = LIFSpike()
        self.shortcut = nn.Sequential(SeqToANNContainer())
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SeqToANNContainer(nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)),
                SeqToANNContainer(nn.BatchNorm2d(self.expansion*planes))
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = SeqToANNContainer(nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False))
        self.bn1 = SeqToANNContainer(nn.BatchNorm2d(64))
        self.lastbn = SeqToANNContainer(nn.BatchNorm1d(512))
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = SeqToANNContainer(nn.Linear(512*block.expansion, num_classes))
        self.act = LIFSpike()
        self.avgpool = SeqToANNContainer(nn.AvgPool2d(4))
        
        self.avgpool1 = nn.AvgPool2d(32)
        self.avgpool2 = nn.AvgPool2d(16)
        self.avgpool3 = nn.AvgPool2d(8)
        self.avgpool4 = nn.AvgPool2d(4)
        self.T = 4
        self.transform1 = transform(64)
        self.transform2 = transform(128)
        self.transform3 = transform(256)
        self.transform4 = transform(512)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    def forward(self, x):
        if len(x.shape) == 4:
            x = add_dimention(x, self.T)
        out = self.act(self.bn1(self.conv1(x)))
        x1 = self.layer1(out)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        out = self.avgpool(x4)
        #print(out.shape)
        #out = out.view(out.size(0), -1)
        out = torch.flatten(out, 2)
       # print(out.shape)
        out = self.linear(out)
        #out = self.lastbn(self.linear(out))
        
        '''test code'''
        # x1 = torch.flatten(self.avgpool1(self.transform1(x1)),1)
        # x2 = torch.flatten(self.avgpool2(self.transform2(x2)),1)
        # x3 = torch.flatten(self.avgpool3(self.transform3(x3)),1)
        # x4 = torch.flatten(self.avgpool4(self.transform4(x4)),1)
        
        # x1 = torch.flatten(self.avgpool1(x1.mean(dim=0)),1)
        # x2 = torch.flatten(self.avgpool2(x2.mean(dim=0)),1)
        # x3 = torch.flatten(self.avgpool3(x3.mean(dim=0)),1)
        # x4 = torch.flatten(self.avgpool4(x4.mean(dim=0)),1)
    
        
        return out.mean(dim=0), [x1.mean(dim=0),x2.mean(dim=0),x3.mean(dim=0),x4.mean(dim=0)]

        #return out.mean(dim=0), [self.transform1(x1), self.transform4(x4)]


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2],  num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


class transform(nn.Module):
    def __init__(self, channel):
        super(transform, self).__init__()
        # self.net = nn.Sequential(
        #     #SeqToANNContainer(nn.Conv2d(channel, channel, 1, 1, 0, bias=False)),
        #     #SeqToANNContainer(nn.BatchNorm2d(channel)),
        #     nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(channel),
        # )
        # 학습 가능한 시간 가중치 설정
        self.T = 4
        self.time_weights = nn.Parameter(torch.ones(self.T), requires_grad=True)

    def forward(self, x):
        T, B, C, H, W = x.shape
        # 가중평균 적용
        normalized_weights = torch.softmax(self.time_weights, dim=0).view(T, 1, 1, 1, 1)  # [T, 1, 1, 1, 1]
        x = x * normalized_weights  # Broadcasting 적용 [T, B, C, H, W]
        x = torch.sum(x, dim=0)  # 시간 차원을 합산 -> [B, C, H, W]
        #x = x.mean(dim=0)
        #x = self.net(x)
        return x


#     def __init__(self, channel):
#         super(transform, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(channel)
#         )
#         # 학습 가능한 시간 가중치 설정
#         self.T = 4
#         self.time_weights = nn.Parameter(torch.ones(self.T), requires_grad=True)

#     def forward(self, x):

#         """
#         x: 입력 텐서 [T, B, C, H, W]
#         """
#         T, B, C, H, W = x.shape

#         x = self.net(x)  # [T, B, C, H, W]

#         # 시간 가중치 정규화 (Softmax 적용)
#         normalized_weights = torch.softmax(self.time_weights, dim=0).view(T, 1, 1, 1, 1)  # [T, 1, 1, 1, 1]
#         # 가중치를 시간 차원에 곱하고 합산 (einsum 대신 broadcasting 사용)
#         x = x * normalized_weights  # Broadcasting 적용 [T, B, C, H, W]
#         x = torch.sum(x, dim=0)  # 시간 차원을 합산 -> [B, C, H, W]
#         # 네트워크를 각 시간 프레임에 적용

#         return x