import torchimport torch

import torch.nn as nnimport torch.nn as nn

import torch.nn.functional as Fimport torch.nn.functional as F



class DoubleConv(nn.Module):

    """(convolution => [BN] => ReLU) * 2"""class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):

    def __init__(self, in_channels, out_channels, mid_channels=None):        super().__init__()

        super().__init__()        self.W_g = nn.Sequential(

        if not mid_channels:            nn.Conv2d(F_g, F_int, kernel_size=1),

            mid_channels = out_channels            nn.BatchNorm2d(F_int)

        self.double_conv = nn.Sequential(        )

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),        self.W_x = nn.Sequential(

            nn.BatchNorm2d(mid_channels),            nn.Conv2d(F_l, F_int, kernel_size=1),

            nn.ReLU(inplace=True),            nn.BatchNorm2d(F_int)

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),        )

            nn.BatchNorm2d(out_channels),        self.psi = nn.Sequential(

            nn.ReLU(inplace=True)            nn.Conv2d(F_int, 1, kernel_size=1),

        )            nn.BatchNorm2d(1),

            nn.Sigmoid()

    def forward(self, x):        )

        return self.double_conv(x)        self.relu = nn.ReLU(inplace=True)



class Down(nn.Module):    def forward(self, g, x):

    """Downscaling with maxpool then double conv"""        g1 = self.W_g(g)

        x1 = self.W_x(x)

    def __init__(self, in_channels, out_channels):        psi = self.relu(g1 + x1)

        super().__init__()        psi = self.psi(psi)

        self.maxpool_conv = nn.Sequential(        return x * psi

            nn.MaxPool2d(2),

            DoubleConv(in_channels, out_channels)

        )class DoubleConv(nn.Module):

    """(convolution => [BN] => ReLU) * 2"""

    def forward(self, x):

        return self.maxpool_conv(x)    def __init__(self, in_channels, out_channels, mid_channels=None):

        super().__init__()

class Up(nn.Module):        if not mid_channels:

    """Upscaling then double conv"""            mid_channels = out_channels

        self.double_conv = nn.Sequential(

    def __init__(self, in_channels, out_channels):            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),

        super().__init__()            nn.BatchNorm2d(mid_channels),

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)            nn.ReLU(inplace=True),

        self.conv = DoubleConv(in_channels, out_channels)            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(out_channels),

    def forward(self, x1, x2):            nn.ReLU(inplace=True)

        x1 = self.up(x1)        )

        

        # Handling cases where input dimensions don't match perfectly    def forward(self, x):

        diffY = x2.size()[2] - x1.size()[2]        return self.double_conv(x)

        diffX = x2.size()[3] - x1.size()[3]



        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,class Down(nn.Module):

                       diffY // 2, diffY - diffY // 2])    """Downscaling with maxpool then double conv"""

        

        x = torch.cat([x2, x1], dim=1)    def __init__(self, in_channels, out_channels):

        return self.conv(x)        super().__init__()

        self.maxpool_conv = nn.Sequential(

class OutConv(nn.Module):            nn.MaxPool2d(2),

    def __init__(self, in_channels, out_channels):            DoubleConv(in_channels, out_channels)

        super(OutConv, self).__init__()        )

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

    def forward(self, x):        return self.maxpool_conv(x)

        return self.conv(x)



class UNet(nn.Module):class Up(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):    """Upscaling with attention then double conv"""

        super(UNet, self).__init__()

        self.n_channels = n_channels    def __init__(self, in_channels, out_channels, bilinear=True):

        self.n_classes = n_classes        super().__init__()

        self.bilinear = bilinear

        # if bilinear, use the normal convolutions to reduce the number of channels

        self.inc = DoubleConv(n_channels, 64)        if bilinear:

        self.down1 = Down(64, 128)            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down2 = Down(128, 256)            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        self.down3 = Down(256, 512)        else:

        factor = 2 if bilinear else 1            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.down4 = Down(512, 1024 // factor)            self.conv = DoubleConv(in_channels, out_channels)

        self.up1 = Up(1024, 512 // factor)

        self.up2 = Up(512, 256 // factor)        # Add attention gate

        self.up3 = Up(256, 128 // factor)        self.attn = AttentionGate(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)

        self.up4 = Up(128, 64)

        self.outc = OutConv(64, n_classes)    def forward(self, x1, x2):

        x1 = self.up(x1)

    def forward(self, x):        # input is CHW

        x1 = self.inc(x)        diffY = x2.size()[2] - x1.size()[2]

        x2 = self.down1(x1)        diffX = x2.size()[3] - x1.size()[3]

        x3 = self.down2(x2)

        x4 = self.down3(x3)        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,

        x5 = self.down4(x4)                        diffY // 2, diffY - diffY // 2])

        x = self.up1(x5, x4)        

        x = self.up2(x, x3)        # Apply attention

        x = self.up3(x, x2)        x2 = self.attn(x1, x2)

        x = self.up4(x, x1)        

        logits = self.outc(x)        # Concatenate

        return logits        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint_sequential([self.inc], 1, self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint_sequential([self.down1], 1, self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint_sequential([self.down2], 1, self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint_sequential([self.down3], 1, self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint_sequential([self.down4], 1, self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint_sequential([self.up1], 1, self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint_sequential([self.up2], 1, self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint_sequential([self.up3], 1, self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint_sequential([self.up4], 1, self.up4)
        self.outc = torch.utils.checkpoint.checkpoint_sequential([self.outc], 1, self.outc)