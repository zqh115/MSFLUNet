import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)  # 30,1,50,30
        x3= self.sigmoid(x2)
        return  x3*x+x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        x1 = avg_out + max_out
        x2 = self.sigmoid(x1)
        return x2*x+x

class FusionBlock(nn.Module):
    def __init__(self,ch_in,k,ch_out):
        super(FusionBlock,self).__init__()
        self.CAB=ChannelAttention(ch_in)
        self.SAB=SpatialAttention()
        self.block1 = nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), dilation=2, padding=(k - 1, k - 1), bias=True,
                              groups=ch_in),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, 2*ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(2*ch_in),
            )
        self.block2=nn.Sequential(
                nn.Conv2d(4*ch_in, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self,x1,x2):
        CA_out=self.CAB(x1)
        SA_out=self.SAB(x2)
        b1_out=self.block1(x1+x2)
        b2_out=self.block2(torch.cat((CA_out,SA_out,b1_out),dim=1))
        return b2_out

class FusionBlock2(nn.Module):
    def __init__(self,ch_in,k,ch_out):
        super(FusionBlock2,self).__init__()
        self.CAB=ChannelAttention(ch_in)
        self.SAB=SpatialAttention()
        self.block1 = nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), dilation=2, padding=(k - 1, k - 1), bias=True,
                              groups=ch_in),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, 2*ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(2*ch_in),
            )
        self.block2=nn.Sequential(
                nn.Conv2d(4*ch_in, ch_out, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self,x1,x2):
        CA_out=self.CAB(x1)
        SA_out=self.SAB(x1)
        b1_out=self.block1(x1+x2)
        b2_out=self.block2(torch.cat((CA_out,SA_out,b1_out),dim=1))
        return b2_out


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
            )
        )
        self.block2 = nn.Sequential(
            nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), dilation=2, padding=(k - 1, k - 1), bias=True,
                              groups=ch_in),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
            )
        )
        self.fusion=FusionBlock(ch_in=ch_in,k=3,ch_out=ch_in)
        self.block3=conv_block(ch_in, ch_out)
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3=self.fusion(x1,x2)
        x4=self.block3(x3)
        return x4


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
        #     nn.GELU(),
        #     nn.BatchNorm2d(ch_in),
        #     nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
        #     nn.GELU(),
        #     nn.BatchNorm2d(ch_out * 4),
        #     nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
        #     nn.GELU(),
        #     nn.BatchNorm2d(ch_out)
        # )
        self.block1 = nn.Sequential(
            nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=True),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
            )
        )
        self.block2 = nn.Sequential(
            nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=True),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
            )
        )
        # self.fusion = FusionBlock2(ch_in=ch_in, k=3, ch_out=ch_out)
        self.conv1=conv_block(ch_in=2*ch_in,ch_out=ch_out)
    def forward(self, x1,x2):
        x3=self.block1(x1)
        x4=self.block2(x2)
        x5=self.conv1(torch.cat((x3,x4),dim=1))
        return x5
        # return self.conv(torch.cat((x1,x2),dim=1))


class MSFLUNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 192, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 3, 3, 3]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(EZNet4_NoRightFusion, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] , ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] , ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] , ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] , ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = self.Up_conv5(x4, d5)

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(x3, d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(x2, d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(x1, d2)
        d1 = self.Conv_1x1(d2)

        return d1

