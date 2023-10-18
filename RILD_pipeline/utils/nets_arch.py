import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# ==============================================
# Home made U-net:
# legacy functions currently supported by torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
# ==============================================


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            # nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            # nn.BatchNorm2d(out_ch, affine=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(.1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv_out(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv_out, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch, affine=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            # nn.BatchNorm2d(out_ch, affine=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up_last(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_last, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv_out(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_filters):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, n_filters)
        self.down1 = down(n_filters, n_filters*2)
        self.down2 = down(n_filters*2, n_filters*4)
        self.down3 = down(n_filters*4, n_filters*8)
        self.down4 = down(n_filters*8, n_filters*16)
        self.up1 = up(n_filters*16 + n_filters*8, n_filters*8)
        self.up2 = up(n_filters*8 + n_filters*4, n_filters*4)
        self.up3 = up(n_filters*4 + n_filters*2, n_filters*2)
        self.up4 = up_last(n_filters*2 + n_filters, n_filters)
        self.outc = outconv(n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output


class UNet_deeper(nn.Module):
    def __init__(self, in_channels, n_classes, n_filters):
        super(UNet_deeper, self).__init__()
        self.inc = inconv(in_channels, n_filters)
        self.down1 = down(n_filters, n_filters*2)
        self.down2 = down(n_filters*2, n_filters*4)
        self.down3 = down(n_filters*4, n_filters*8)
        self.down4 = down(n_filters*8, n_filters*16)
        self.down5 = down(n_filters * 16, n_filters * 32)
        self.up1 = up(n_filters * 32 + n_filters * 16, n_filters * 16)
        self.up2 = up(n_filters*16 + n_filters*8, n_filters*8)
        self.up3 = up(n_filters*8 + n_filters*4, n_filters*4)
        self.up4 = up(n_filters*4 + n_filters*2, n_filters*2)
        self.up5 = up_last(n_filters*2 + n_filters, n_filters)
        self.outc = outconv(n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        output = self.outc(x)
        return output


class UNet_small(nn.Module):
    def __init__(self, in_channels, n_classes, n_filters):
        super(UNet_small, self).__init__()
        self.inc = inconv(in_channels, n_filters)
        self.down1 = down(n_filters, n_filters*2)
        self.down2 = down(n_filters*2, n_filters*4)
        self.down3 = down(n_filters*4, n_filters*8)
        self.up2 = up(n_filters*8 + n_filters*4, n_filters*4)
        self.up3 = up(n_filters*4 + n_filters*2, n_filters*2)
        self.up4 = up_last(n_filters*2 + n_filters, n_filters)
        self.outc = outconv(n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output

class UNet_smaller(nn.Module):
    def __init__(self, in_channels, n_classes, n_filters):
        super(UNet_smaller, self).__init__()
        self.inc = inconv(in_channels, n_filters)
        self.down1 = down(n_filters, n_filters*2)
        self.down2 = down(n_filters*2, n_filters*4)
        # self.down3 = down(n_filters*4, n_filters*8)
        # self.up2 = up(n_filters*8 + n_filters*4, n_filters*4)
        self.up3 = up(n_filters*4 + n_filters*2, n_filters*2)
        self.up4 = up_last(n_filters*2 + n_filters, n_filters)
        self.outc = outconv(n_filters, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        #
        # x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output