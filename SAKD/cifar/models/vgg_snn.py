import math

import torch
import torch.nn as nn
import torch.nn.init as init
from .layers import *

__all__ = [
    'VGG', 'vgg16',
]


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, cfg, num_classes=10, batch_norm=True, width_mult=1, in_c=2, **lif_parameters):
        super(VGG, self).__init__()

        self.features, out_c = make_layers(cfg, batch_norm, width_mult, in_c, **lif_parameters)
        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((7, 7)))
        self.classifier = nn.Sequential(
            # nn.Dropout(0.25),
            SeqToANNContainer(nn.Linear(out_c * 7 * 7, 4096)),
            LIFSpike(),
            nn.Dropout(0.5),
            SeqToANNContainer(nn.Linear(4096, 4096)),
            LIFSpike(),
            nn.Dropout(0.5),
            SeqToANNContainer(nn.Linear(4096, num_classes)),
        )

        self.T = 4
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.add_dim = lambda x: add_dimention(x, self.T)
        
        self.transform1 = transform(64)
        self.transform2 = transform(128)
        self.transform3 = transform(256)
        self.transform4 = transform(512)
        self.transforms = [self.transform1, self.transform2, self.transform3, self.transform4]

    def forward(self, x):
        feature_maps = []
        if len(x.shape) == 4:
            x = add_dimention(x, self.T)

        for feature in self.features:
            x = feature(x)
            if isinstance(feature, SeqToANNContainer) and isinstance(feature.module, torch.nn.AvgPool2d):
                feature_maps.append(x)
                #print('feature : ', x.shape)
        x = self.avgpool(x)
        feature_output = torch.flatten(x, 1) if len(x.shape) == 4 else torch.flatten(x, 2)
        output = self.classifier(feature_output)
        
        #output = self.lastbn(output)
        if True:
            length = min(len(self.transforms), len(feature_maps))
            result = [self.transforms[i](feature_maps[i]) for i in range(length)]
            #for r in result:
                #print('after : ', r.shape)
            return output.mean(dim=0), result
        else:
            return output.mean(dim=0), feature_maps


def make_layers(cfg, batch_norm=True, width_mult=1, in_c=2, **lif_parameters):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = SeqToANNContainer(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))

            lif = LIFSpike(**lif_parameters)

            if batch_norm:
                bn = tdBatchNorm(int(v))
                layers += [conv2d, bn, lif]
            else:
                layers += [conv2d, lif]

            in_channels = v
    return nn.Sequential(*layers), in_channels


cfg = {
    #"5": [64, 'M', 128, 128, 'M'], #5
    "9": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'], #9
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], #11
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, 512], #13
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], #16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], #17
}


# def vgg5(*args, **kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG(cfg['5'], *args, **kwargs)

def vgg9(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['9'], *args, **kwargs)

def vgg11(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['A'], *args, **kwargs)

def vgg13(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['B'], *args, **kwargs)

def vgg16(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['D'], *args, **kwargs)

def vgg19(*args, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['E'], *args, **kwargs)

class transform(nn.Module):
    def __init__(self, channel):
        super(transform, self).__init__()
        # self.net = nn.Sequential(
        #     SeqToANNContainer(nn.Conv2d(channel, channel, 1, 1, 0, bias=False)),
        #     SeqToANNContainer(nn.BatchNorm2d(channel)),
        #     #nn.Linear(channel*8*8, channel*16),  # Linear projection
        #     #nn.BatchNorm1d(channel*16, eps=0.0, affine=False),
        # )
        # 학습 가능한 시간 가중치 설정
        self.T = 4
        #self.time_weights = nn.Parameter(torch.ones(self.T), requires_grad=True)

    def forward(self, x):
        # normalized_weights = torch.softmax(self.time_weights, dim=0).view(self.T, 1, 1, 1, 1)  # [T, 1, 1, 1, 1]
        # x = x * normalized_weights  # Broadcasting 적용 [T, B, C, H, W]
        # x = torch.sum(x, dim=0)  # 시간 차원을 합산 -> [B, C, H, W]
        x = x.mean(dim=0)
        return x

# class transform(nn.Module):
#     def __init__(self, channel):
#         super(transform, self).__init__()
#         self.net = nn.Sequential(
#             SeqToANNContainer(nn.Conv2d(channel, channel, 1, 1, 0, bias=False)),
#             SeqToANNContainer(nn.BatchNorm2d(channel)),
#             #nn.Linear(channel*8*8, channel*16),  # Linear projection
#             #nn.BatchNorm1d(channel*16, eps=0.0, affine=False),
#         )

#     def forward(self, x):
#         #print(x.shape)
#         x = self.net(x)
#         x = torch.mean(x, dim=0)
#         return x

if __name__ == '__main__':
    model = vgg16(num_classes=10, width_mult=1)
    model.T = 2
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
    y.sum().backward()
    print(y.shape)