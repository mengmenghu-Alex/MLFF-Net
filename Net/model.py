import torch
import torch.nn as nn
from Net.base_net import *
import torchvision.models as models

def swish(x):
    return x * x.sigmoid()

def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)

class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)

def _make_divisible(v, divisor=8, min_value=None): 
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock1(nn.Module):
    def __init__(self):
        super(ConvBlock1, self).__init__()
        self.DW = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, groups=16, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(16)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(32)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock2(nn.Module):
    def __init__(self):
        super(ConvBlock2, self).__init__()
        self.DW = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, groups=32, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(32)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(64)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock3(nn.Module):
    def __init__(self):
        super(ConvBlock3, self).__init__()
        self.DW = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, groups=128, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(128)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(64)

    def forward(self, x):
        a = self.HS(self.BN(self.DW(x)))
        a = self.HS(self.BNN(self.PW(a)))
        return a

class ConvBlock4(nn.Module):
    def __init__(self):
        super(ConvBlock4, self).__init__()
        self.DW = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, groups=128, padding=1, bias=False)
        self.BN = nn.BatchNorm2d(128)
        self.HS = HardSwish()
        self.PW = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.BNN = nn.BatchNorm2d(64)
        self.SE = SELayer(128, 128)

    def forward(self, x):

        a = self.HS(self.BN(self.DW(x)))
        a = self.SE(a)
        a = self.HS(self.BNN(self.PW(a)))
        return a

class VGG19Extractor(nn.Module):
    def __init__(self):
        super(VGG19Extractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.features = vgg19.features
        self.layer1 = nn.Sequential(*self.features[:4]) 
        self.layer2 = nn.Sequential(*self.features[4:8])  
        self.layer3 = nn.Sequential(*self.features[8:16])
        self.layer4 = nn.Sequential(*self.features[16:24])  

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4 

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_channels)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn7x7 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        conv7x7 = self.conv7x7(x)
        out = torch.cat([conv3x3, conv7x7], dim=1)
        
        return out


class MLLF(nn.Module):
    def __init__(self, num_layers=3):
        super(MLLF, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False) 
        
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_r1 = MaxPooling2D()
        self.maxpool_r2 = MaxPooling2D()
        self.deconv_r1 = ConvTranspose2D(128, 128)
        self.concat_r1 = Concat()
        self.deconv_r2 = ConvTranspose2D(64, 128)
        self.concat_r2 = Concat()
        self.out = nn.Sigmoid()

        self.block1 = ConvBlock1()
        self.block2 = ConvBlock2()
        self.block3 = ConvBlock3()
        self.block4 = ConvBlock4()
        self.multi_scale_0 = MultiScaleConv(576,64)
        self.multi_scale_1 = MultiScaleConv(416,64)
        self.multi_scale_2 = MultiScaleConv(272,64)
        self.vgg19_extractor = VGG19Extractor()

    def forward(self, x):
        vgg_x2, vgg_x3, vgg_x4 = self.vgg19_extractor(x)
        # 3->16
        x = self.input(x)
        '''
        vgg_x2 128
        vgg_x3 256
        vgg_x4 512
        '''
        maxpool_r1 = self.maxpool_r1(x)
        # 16->32
        x1 = self.block1(maxpool_r1)

        maxpool_r2 = self.maxpool_r2(x1)
        # 32->64
        x2 = self.block2(maxpool_r2)
        x2 = self._resize_and_concat(x2, vgg_x4)
        x2_fusion = self.multi_scale_0(x2)

        # 64->128
        deconv_r1 = self.deconv_r1(x2_fusion)
        # 128->128+32=160
        concat_r1 = self.concat_r1(x1, deconv_r1)
        concat_r1 = self._resize_and_concat(concat_r1, vgg_x3)
        d1_fusion = self.multi_scale_1(concat_r1)
        # 160->64
        x3 = self.block3(d1_fusion)
        
        # 64->128
        deconv_r2 = self.deconv_r2(x3)
        # 128->128+16=144
        concat_r2 = self.concat_r2(x, deconv_r2)
        concat_r2 = self._resize_and_concat(concat_r2, vgg_x2)
        d2_fusion = self.multi_scale_2(concat_r2)
        # 144->64
        x4 = self.block4(d2_fusion)
        # 64->3
        out = self.output(x4)
        
        out = self.out(out)
        return out
    
    def _resize_and_concat(self, decoder_output, vgg_output):
        _, _, h, w = decoder_output.size()

        vgg_resized = nn.functional.interpolate(vgg_output, size=(h, w), mode='bilinear', align_corners=False)

        return torch.cat([decoder_output, vgg_resized], dim=1)  
