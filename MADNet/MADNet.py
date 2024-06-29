# -------------------------------------------------------------
# File: MADNet.py
# Author: Qiang Li
# Date of Completion: June 27, 2024
# Description: Neural network model
# -------------------------------------------------------------
# Input/Output Information (IO):
# Input: /
# Output: /
# -------------------------------------------------------------
from MADNet.MADNet_parts import *

class MADNet(nn.Module):
    def __init__(self, in_ch, out_ch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch

        # in
        self.inc = nn.Sequential(nn.Conv2d(in_ch, 16, kernel_size=1, padding=0, bias=False),
                                 nn.BatchNorm2d(16),
                                 nn.ELU(inplace=False),
                                 # nn.Conv2d(16, 16, kernel_size=1, padding=0, bias=False),
                                 # nn.BatchNorm2d(16),
                                 # nn.ReLU(inplace=False),
                                 SELayer(16),
                                 SpatialAttention()
                                 )
        # down1
        self.down1 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.ELU(inplace=False),
                                   # nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
                                   # nn.BatchNorm2d(32),
                                   # nn.ReLU(inplace=False),
                                   # SELayer(32),
                                   SpatialAttention()
                                   )

        # down2
        self.down2 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ELU(inplace=True),
                                   # nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
                                   # nn.BatchNorm2d(64),
                                   # nn.ReLU(inplace=False),
                                   # SELayer(64),
                                   SpatialAttention()
                                   )
        # down3
        self.down3 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ELU(inplace=False),
                                   # nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, bias=False),
                                   # nn.BatchNorm2d(128),
                                   # nn.ReLU(inplace=False),
                                   # SELayer(128),
                                   SpatialAttention()
                                   )
        # down4
        self.down4 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ELU(inplace=False),
                                   # nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=False),
                                   # nn.BatchNorm2d(256),
                                   # nn.ReLU(inplace=False),
                                   # SELayer(256),
                                   SpatialAttention()
                                   )
        # down5
        self.down5 = nn.Sequential(nn.MaxPool2d(2),
                                   nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ELU(inplace=False),
                                   # nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias=False),
                                   # nn.BatchNorm2d(512),
                                   # nn.ReLU(inplace=False),
                                   SELayer(512),
                                   # SpatialAttention()
                                   )
        # down6
        self.down6 = nn.Sequential(
                                    nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ELU(inplace=False),
                                    # nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2, bias=False),
                                    # nn.BatchNorm2d(512),
                                    # nn.ReLU(inplace=False),
                                    SELayer(512),
                                    # SpatialAttention()
                                    )
        self.up1 = Up1(512, 256)
        self.up2 = Up1(256, 128)
        self.up3 = Up2(128, 64)
        self.up4 = Up1(64, 32)
        self.up5 = Up1(32, 16)
        self.outc = OutConv(16, out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x4)
        x10 = self.up3(x9, x3)
        x11 = self.up4(x10, x2)
        x12 = self.up5(x11, x1)
        x13 = self.outc(x12)
        return x13

    def use_checkpointing(self):
        pass

