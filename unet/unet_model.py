""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x = torch.randn(size=(1, 1, 512, 512))
        # print('x.shape: ', x.shape)
        # x.shape:  torch.Size([1, 1, 512, 512])
        x1 = self.inc(x)
        # print('After inc.shape: ', x1.shape)
        # After inc.shape:  torch.Size([1, 64, 512, 512])
        x2 = self.down1(x1)
        # print('After down1.shape: ', x2.shape)
        # After down1.shape:  torch.Size([1, 128, 256, 256])
        x3 = self.down2(x2)
        # print('After down2.shape: ', x3.shape)
        # After down2.shape:  torch.Size([1, 256, 128, 128])
        x4 = self.down3(x3)
        # print('After down3.shape: ', x4.shape)
        # After down3.shape:  torch.Size([1, 512, 64, 64])
        x5 = self.down4(x4)
        # print('After down4.shape: ', x5.shape)
        # After down4.shape:  torch.Size([1, 512, 32, 32])
        x = self.up1(x5, x4)
        # print('After up1.shape: ', x.shape)
        # After up1.shape:  torch.Size([1, 256, 64, 64])
        x = self.up2(x, x3)
        # print('After up2.shape: ', x.shape)
        # After up2.shape:  torch.Size([1, 128, 128, 128])
        x = self.up3(x, x2)
        # print('After up3.shape: ', x.shape)
        # After up3.shape:  torch.Size([1, 64, 256, 256])
        x = self.up4(x, x1)
        # print('After up4.shape: ', x.shape)
        # After up4.shape:  torch.Size([1, 64, 512, 512])
        logits = self.outc(x)
        # print('After logits.shape: ', x.shape)
        # After logits.shape:  torch.Size([1, 64, 512, 512])
        return logits
