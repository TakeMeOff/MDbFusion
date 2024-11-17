import torch
import torch.nn as nn
from typing import Union
base_channels = 32
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 0)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = int(in_batch / r ** 2), int(
        in_channel), r * in_height, r * in_width
    x1 = x[0:out_batch, :, :, :] / 2
    x2 = x[out_batch:out_batch * 2, :, :, :] / 2
    x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2
    x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class DiaBlock(nn.Module):
    def __init__(self, input_channels, out_channels, dia, need_expand=False):
        super(DiaBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=int(out_channels / 2), kernel_size=3,
                               stride=1, dilation=dia[0], padding=dia[0])
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=int(out_channels / 2), kernel_size=3,
                               stride=1, dilation=dia[1], padding=dia[1])
        self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=int(out_channels / 2), kernel_size=3,
                               stride=1, dilation=dia[2], padding=dia[2])
        self.conv4 = nn.Conv2d(in_channels=input_channels, out_channels=int(out_channels / 2), kernel_size=3,
                               stride=1, dilation=dia[3], padding=dia[3])
        self.conv = nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3,
                              stride=1, dilation=1, padding=1)
        self.elu = nn.ELU(alpha=1.0, inplace=True)
        self.need_expand = need_expand
        self.expand = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.elu(self.conv1(x))
        x2 = self.elu(self.conv2(x))
        x3 = self.elu(self.conv3(x))
        x4 = self.elu(self.conv4(x))
        temp = torch.cat((x1, x2, x3, x4), dim=1)
        temp = self.elu(self.conv(temp))
        if self.need_expand:
            x = self.expand(x)
        return x + temp


class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        self.downconv1 = nn.Conv2d(in_channels=input_channels, out_channels=1 * base_channels, kernel_size=7, stride=1,
                                   padding=0)
        self.downconv2 = nn.Conv2d(in_channels=1 * base_channels, out_channels=2 * base_channels, kernel_size=3,
                                   stride=2, padding=1)
        self.block1 = DiaBlock(input_channels=2 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8],
                               need_expand=True)
        self.block2 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block3 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block4 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block5 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block6 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block7 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block8 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block9 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 2, 4, 8])
        self.block10 = DiaBlock(input_channels=4 * base_channels, out_channels=4 * base_channels, dia=[1, 3, 5, 7])
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.elu = nn.ELU(alpha=1.0, inplace=True)
        self.reflecpad = nn.ReflectionPad2d(3)

    def forward(self, x):
        temp = []
        x = self.reflecpad(x)
        x = self.elu(self.downconv1(x))
        temp.append(x)
        x = self.elu(self.downconv2(x))
        temp.append(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        x = torch.cat((x, temp[1]), dim=1)
        x = self.upsample(x)
        x = torch.cat((x, temp[0]), dim=1)
        x = self.elu(x)

        '''
        x = self.upsample(x)
        x = torch.cat((x, temp[1]), dim=1)
        x = self.upsample(x)
        x = torch.cat((x, temp[0]), dim=1)
        '''
        return x


class Decoder(nn.Module):
    def __init__(self, output_channels):
        super(Decoder, self).__init__()
        self.decoder1 = nn.Conv2d(in_channels=7 * base_channels, out_channels=base_channels, kernel_size=3, stride=1,
                                  padding=1)
        self.decoder2 = nn.Conv2d(in_channels=base_channels, out_channels=output_channels, kernel_size=3, stride=1,
                                  padding=1)
        self.elu = nn.ELU(alpha=1.0, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dwt = DWT()
        self.iwt = IWT()
        self.wconv1 = nn.Conv2d(7 * base_channels, 64, kernel_size=9, padding=9 // 2)
        self.wconv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.wconv3 = nn.Conv2d(32, output_channels, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        out1 = self.elu(self.decoder1(x))
        out1 = self.decoder2(out1)

        out2 = self.dwt(x)
        out2 = self.relu(self.wconv1(out2))
        out2 = self.relu(self.wconv2(out2))
        out2 = self.wconv3(out2)
        out2 = self.iwt(out2)
        return out1 + out2


class Net(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Net, self).__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.decoder = Decoder(output_channels=output_channels)

    def forward(self, x):
        input = x
        input = self.encoder(input)
        input = self.decoder(input)
        return x + input


class FusionNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FusionNet, self).__init__()
        self.encoder = Encoder(input_channels=input_channels)
        self.decoder = Decoder(output_channels=output_channels)
        self.w_vi1 = nn.Conv2d(in_channels=7 * base_channels, out_channels=7 * base_channels, stride=1, padding=1,
                               kernel_size=3, groups=7 * base_channels)
        self.w_vi2 = nn.Conv2d(in_channels=7 * base_channels, out_channels=7 * base_channels, stride=1, padding=1,
                               kernel_size=3, groups=7 * base_channels)
        self.w_if1 = nn.Conv2d(in_channels=7 * base_channels, out_channels=7 * base_channels, stride=1, padding=1,
                               kernel_size=3, groups=7 * base_channels)
        self.w_if2 = nn.Conv2d(in_channels=7 * base_channels, out_channels=7 * base_channels, stride=1, padding=1,
                               kernel_size=3, groups=7 * base_channels)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.y_conv1 = nn.Conv2d(in_channels=7 * base_channels, out_channels=64, stride=1, padding=1, kernel_size=3)
        self.y_conv2 = nn.Conv2d(in_channels=64, out_channels=32, stride=1, padding=1, kernel_size=3)
        self.y_conv3 = nn.Conv2d(in_channels=32, out_channels=1, stride=1, padding=1, kernel_size=3)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ELU(alpha=1.0, inplace=True)

    def forward(self, img_vi, img_if):
        # img_vi: [B, 3, H, W]
        # img_if: [B, 1x3, H, W]

        '''
        a = self.encoder(img_vi)
        a = self.decoder(a)
        return a + img_vi
        '''

        # feature extraction
        fi_vi = self.encoder(img_vi)
        fi_if = self.encoder(img_if)

        # weight calculate
        weight_vi = self.relu(self.w_vi1(fi_vi))
        weight_vi = self.relu(self.w_vi2(weight_vi))
        weight_vi = self.GAP(weight_vi)
        weight_if = self.relu(self.w_if1(fi_if))
        weight_if = self.relu(self.w_if2(weight_if))
        weight_if = self.GAP(weight_if)
        weight = torch.cat((weight_vi.unsqueeze(0), weight_if.unsqueeze(0)), dim=0)
        weight = self.softmax(weight)
        weight_vi = weight[0]
        weight_if = weight[1]

        # fused feature
        feature_fused = weight_if * fi_if + weight_vi * fi_vi
        # feature_fused = 0.5 * fi_if + 0.5 * fi_vi

        # fused y channel
        fused_y = self.elu(self.y_conv1(feature_fused))
        fused_y = self.elu(self.y_conv2(fused_y))
        fused_y = self.elu(self.y_conv3(fused_y))

        # deblur residual img
        residual = self.decoder(feature_fused)

        # final result
        # fused_blur_img = torch.cat((fused_y, img_vi[:, 1:, :, :]), dim=1)
        # fused_deblur_img = fused_blur_img + residual
        fused_deblur_img = img_vi + residual

        return fused_deblur_img


if __name__ == '__main__':
    device = "cuda"
    model = FusionNet(3, 3)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    image = torch.rand(2, 3, 256, 256)
    image = image.to(device)
    encoder = model.to(device)
    with torch.no_grad():
        output = model(image, image).to(device)
        print(output.shape)
