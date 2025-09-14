import torch
import torch.nn as nn
from .layers import *

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=16, dilation=1, T=4):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 16:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SeqToANNContainer(conv3x3(inplanes, planes, stride))
        self.bn1 = SeqToANNContainer(nn.BatchNorm2d(planes))

        self.conv2 = SeqToANNContainer(conv3x3(planes, planes))
        self.bn2 = SeqToANNContainer(nn.BatchNorm2d(planes))
        self.downsample = downsample
        self.stride = stride
        self.shortcut = nn.Sequential(SeqToANNContainer())
        if stride != 1 or inplanes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SeqToANNContainer(nn.Conv2d(inplanes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)),
                SeqToANNContainer(nn.BatchNorm2d(self.expansion*planes))
            )

        self.spike1 = LIFSpike()
        self.spike2 = LIFSpike()

        self.T = T

    def forward(self, x):
        out = self.spike1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.spike2(out)
        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=16, replace_stride_with_dilation=None,
                 norm_layer=None, T=4,step=3,channel=[64,128,256,512]):
        super(ResNet, self).__init__()

        self.step = step
        self.T = 4

        self.inplanes = channel[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = SeqToANNContainer(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False))
        self.bn1 = SeqToANNContainer(nn.BatchNorm2d(self.inplanes))
        self.layer1 = self._make_layer(block, channel[1], layers[0], T=2)
        self.layer2 = self._make_layer(block, channel[2], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], T=4)
        self.layer3 = self._make_layer(block, channel[3], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], T=6)

        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = SeqToANNContainer(nn.Linear(channel[3] * block.expansion, 256))
        self.fc_ = SeqToANNContainer(nn.Linear(256, num_classes))

        self.transform1 = transform(channel[1])
        self.transform2 = transform(channel[2])
        self.transform3 = transform(channel[3])

        self.spike = LIFSpike()

        self.spike_ = LIFSpike()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, T=4):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     downsample = nn.Sequential(
        #         SeqToANNContainer(nn.Conv2d(self.inplanes, planes * block.expansion, stride)),
        #         SeqToANNContainer(nn.BatchNorm2d(planes * block.expansion)),
        #     )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, T=T))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                T=T + i))

        return nn.Sequential(*layers)
        #return nn.ModuleList(layers)

    # def _forward_impl(self, x):
    def forward(self, x):
        if len(x.shape) == 4:
            x = add_dimention(x, self.T)
        feature = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.spike(x)


        x = self.layer1(x)
        #feature.append(self.transform1(x))

        x = self.layer2(x)
        #feature.append(self.transform2(x))

        x = self.layer3(x)
        #feature.append(self.transform3(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 2)

        out = self.fc(x)
        out = self.spike_(out)
        out = self.fc_(out)
        out = torch.mean(out,dim=0)

        return out, feature



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet19_(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet19', BasicBlock, [3, 3, 2], pretrained, progress,
                   **kwargs)

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
