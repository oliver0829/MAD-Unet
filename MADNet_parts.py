import torch
import torch.nn as nn


class Up1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                     output_padding=1, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x1, x2):
        x3 = self.up(x1)
        x4 = torch.cat([x2, x3], dim=1)
        x5 = self.conv(x4)
        return x5


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1,
                                     output_padding=1, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x1, x2):
        x3 = self.up(x1)
        x4 = torch.cat([x2, x3], dim=1)
        x5 = self.conv(x4)
        return x5


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


if __name__ == '__main__':
    layer = SpatialAttention()
    input = torch.randn(5, 10, 100, 100)
    out = layer(input)
    print(out.shape)
