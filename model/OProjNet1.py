import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.io as scio
from math import sqrt
from numpy import clip
from torchvision.transforms import ToPILImage
import torch.nn.init as init



class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.factor = opt.scale
        self.angRes = opt.angular_num
        self.channel = opt.channel

        self.highF = nn.Conv2d(1, self.channel, kernel_size=3, stride=1, padding=1)
        self.midF = nn.Conv2d(1, self.channel, kernel_size=3, stride=2, padding=1)
        self.lowF = nn.Conv2d(1, self.channel, kernel_size=3, stride=4, padding=1)

        self.highResB1 = ResidualBlock_noBN(self.channel)
        self.midResB1 = ResidualBlock_noBN(self.channel)
        self.lowResB1 = ResidualBlock_noBN(self.channel)

        self.conv1 = nn.Conv2d(self.channel*2, self.channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.channel*2, self.channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.channel*3, self.channel, kernel_size=3, stride=1, padding=1)

        self.UpProjectionLow1 = UpProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.DownProjectionLow1 = DownProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.UpProjectionMid1 = UpProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.DownProjectionMid1 = DownProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.UpProjectionMid2 = UpProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.DownProjectionMid2 = DownProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.UpProjectionHigh1 = UpProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.DownProjectionHigh1 = DownProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.UpProjectionHigh2 = UpProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.DownProjectionHigh2 = DownProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.UpProjectionHigh3 = UpProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.DownProjectionHigh3 = DownProjection(nf=self.channel, ang=self.angRes*self.angRes)
        self.FBM = FBM(self.channel)
        self.UpSample = Upsample(self.channel, self.factor)

    def forward(self, lf_lr):
        B, an2, H, W = lf_lr.size()  ### N=1 ###
        
        lowFea = self.lowF(lf_lr.view(-1, 1, H, W))
        midFea = self.midF(lf_lr.view(-1, 1, H, W))
        highFea = self.highF(lf_lr.view(-1, 1, H, W))
        highFea = highFea - F.interpolate(midFea, scale_factor=2, mode='bilinear', align_corners=False)
        midFea = midFea - F.interpolate(lowFea, scale_factor=2, mode='bilinear', align_corners=False)

        lowFea = self.DownProjectionLow1(self.UpProjectionLow1(lowFea.view(B, -1, self.channel, H//4, W//4))).view(-1, self.channel, H//4, W//4)
        
        midFea1 = self.DownProjectionMid1(self.UpProjectionMid1(midFea.view(B, -1, self.channel, H//2, W//2)))
        midFea2 = torch.cat((midFea1.view(-1, self.channel, H//2, W//2), F.interpolate(lowFea, scale_factor=2, mode='bilinear', align_corners=False)),1)
        midFea2 = self.conv1(midFea2)
        midFea3 = self.DownProjectionMid2(self.UpProjectionMid2(midFea2.view(B, -1, self.channel, H//2, W//2))).view(-1, self.channel, H//2, W//2)
        
        highFea1 = self.DownProjectionHigh1(self.UpProjectionHigh1(highFea.view(B, -1, self.channel, H, W)))
        highFea2 = self.DownProjectionHigh2(self.UpProjectionHigh2(highFea1))
        highFea3 = torch.cat((highFea2.view(-1, self.channel, H, W), F.interpolate(midFea3, scale_factor=2, mode='bilinear', align_corners=False)),1)
        highFea3 = self.conv2(highFea3)
        highFea4 = self.DownProjectionHigh3(self.UpProjectionHigh3(highFea3.view(B, -1, self.channel, H, W))).view(-1, self.channel, H, W)
        
        lowFea_final = F.interpolate(lowFea.view(-1, self.channel, H//4, W//4), scale_factor=4, mode='bilinear', align_corners=False)
        midFea_final = F.interpolate(midFea3.view(-1, self.channel, H//2, W//2), scale_factor=2, mode='bilinear', align_corners=False)

        fea = self.conv3(torch.cat((highFea4,midFea_final,lowFea_final), 1))
        fea = self.FBM(fea.view(B, -1, self.channel, H, W))
        out_sv = self.UpSample(fea)
        # print(out_sv.shape)

        out = torch.squeeze(out_sv, 2)+ lf_lr

        return out


def stack2jigsaw(x, angRes):
    """
    Transform stacked light field to a jigsaw one
    :param x: [N, UV, X, Y]
    :param angRes: angular resolution
    :return: [N, 1, UX, VY]
    """
    N, X, Y = x.shape[0], x.shape[2], x.shape[3]

    x = x.view(N, angRes, angRes, X, Y) # [N, U, V, X, Y]
    x = x.permute([0, 1, 3, 2, 4]).contiguous() # [N, U, X, V, Y]
    x = x.view(N, angRes * X, angRes * Y) # [N, UX, VY]
    x = x.unsqueeze(1) # [N, 1, UX, VY]
    return x

def jigsaw2stack(x, angRes):
    """
    Transform jigsawed light field to a stacked one
    :param x: [N, 1, UX, VY]
    :param angRes: angular resolution
    :return: [N, UV, X, Y]
    """
    N, UX, VY = x.shape[0], x.shape[2], x.shape[3]
    X = UX // angRes
    Y = VY // angRes

    x = x.squeeze(1).view(N, angRes, X, angRes, Y) # [N, U, X, V, Y]
    x = x.permute([0, 1, 3, 2, 4]).contiguous() # [N, U, V, X, Y]
    x = x.view(N, -1, X, Y) # [N, UV, X, Y]
    return x

class Upsample(nn.Module):
    def __init__(self, channel, factor):
        super(Upsample, self).__init__()
        self.upsp = nn.Sequential(
            nn.Conv2d(channel, channel*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.PixelShuffle(factor),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        out = self.upsp(x)
        _, _, H, W = out.shape
        out = out.contiguous().view(b, n, -1, H, W)
        return out


class FeaExtract(nn.Module):
    def __init__(self, channel):
        super(FeaExtract, self).__init__()
        self.FEconv = nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.FERB_1 = ResASPP(channel)
        self.FERB_2 = ResidualBlock_noBN(channel)
        self.FERB_3 = ResASPP(channel)
        self.FERB_4 = ResidualBlock_noBN(channel)

    def forward(self, x_mv):
        b, n, h, w = x_mv.shape
        x_mv = x_mv.contiguous().view(b * n, -1, h, w)
        intra_fea_0 = self.FEconv(x_mv)
        intra_fea = self.FERB_1(intra_fea_0)
        intra_fea = self.FERB_2(intra_fea)
        intra_fea = self.FERB_3(intra_fea)
        intra_fea = self.FERB_4(intra_fea)
        # _, c, h, w = intra_fea.shape
        # intra_fea = intra_fea.unsqueeze(1).contiguous().view(b, -1, c, h,
                                                            #  w)  # .permute(0,2,1,3,4)  # intra_fea:  B, N, C, H, W
        return intra_fea

class RB(nn.Module):
    '''
    Residual Block
    '''
    def __init__(self, channel):
        super(RB, self).__init__()
        self.conv01 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv02 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        buffer = self.conv01(x)
        buffer = self.lrelu(buffer)
        buffer = self.conv02(buffer)
        return buffer + x


class SELayer(nn.Module):
    '''
    Channel Attention
    '''

    def __init__(self, out_ch, g=16):
        super(SELayer, self).__init__()
        self.att_c = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // g, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // g, out_ch, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fm):
        ##channel
        fm_pool = F.adaptive_avg_pool2d(fm, (1, 1))
        att = self.att_c(fm_pool)
        fm = fm * att
        return fm


class FBM(nn.Module):
    '''
    Feature Blending 
    '''

    def __init__(self, channel):
        super(FBM, self).__init__()
        self.FERB_1 = RB(channel)
        self.FERB_2 = RB(channel)
        self.FERB_3 = RB(channel)
        self.FERB_4 = RB(channel)
        self.att1 = SELayer(channel)
        self.att2 = SELayer(channel)
        self.att3 = SELayer(channel)
        self.att4 = SELayer(channel)

    def forward(self, x):
        b, n, c, h, w = x.shape
        buffer_init = x.contiguous().view(b * n, -1, h, w)
        buffer_1 = self.att1(self.FERB_1(buffer_init))
        buffer_2 = self.att2(self.FERB_2(buffer_1))
        buffer_3 = self.att3(self.FERB_3(buffer_2))
        buffer_4 = self.att4(self.FERB_4(buffer_3))
        buffer = buffer_4.contiguous().view(b, n, -1, h, w)
        return buffer


def ChannelSplit(input):
    _, C, _, _ = input.shape
    c = C // 4
    output_1 = input[:, :c, :, :]
    output_2 = input[:, c:, :, :]
    return output_1, output_2


class ResASPP(nn.Module):
    def __init__(self, channel):
        super(ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1,
                                              dilation=1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=2,
                                              dilation=2, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=4,
                                              dilation=4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv2d(channel * 3, channel, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


class D3ResASPP(nn.Module):
    def __init__(self, channel):
        super(D3ResASPP, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1), dilation=(2, 1, 1),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_3 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(4, 1, 1), dilation=(4, 1, 1),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_t = nn.Conv3d(channel * 3, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                                dilation=(1, 1, 1))

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1


def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st


def FormOutput(intra_fea):
    b, n, c, h, w = intra_fea.shape
    angRes = int(sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(intra_fea[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class ResidualBlock_noBN_CCA(nn.Module):#############################################3 new
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN_CCA, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.conv_du = nn.Sequential(
            nn.Conv2d(nf, 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, nf, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # initialization
        initialize_weights([self.conv1, self.conv_du], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.conv2(out)
        out = self.contrast(out)+self.avg_pool(out)
        out_channel = self.conv_du(out)
        out_channel = out_channel*out
        out_last = out_channel+identity

        return out_last

class DownProjection(torch.nn.Module):
    def __init__(self, nf=32, ang=25):
        super(DownProjection, self).__init__()
        self.down_conv1 = DownModule(nf, ang)
        self.down_conv2 = UpModule(nf, ang)
        self.down_conv3 = DownModule(nf, ang)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class UpProjection(torch.nn.Module):
    def __init__(self, nf=32, ang=25):
        super(UpProjection, self).__init__()
        self.up_conv1 = UpModule(nf, ang)
        self.up_conv2 = DownModule(nf, ang)
        self.up_conv3 = UpModule(nf, ang)      

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class UpModule(torch.nn.Module):
    def __init__(self, nf=32, ang=25):
        super(UpModule, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)     
        self.ProgressiveFusion_Block = ProgressiveFusion_Block(nf, ang)

    def forward(self, x):
        # x: torch.Size([2, 25, 32, 64, 64])
        B,an2,C,H,W = x.shape
        x = self.ProgressiveFusion_Block(x)
        x = x.view(-1,C,H,W)
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = self.conv(x).view(B,an2,C,H*2,W*2)
        return x

class DownModule(torch.nn.Module):
    def __init__(self, nf=32, ang=25):
        super(DownModule, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 5, 2, 2, bias=True)     
        self.ProgressiveFusion_Block = ProgressiveFusion_Block(nf, ang)

    def forward(self, x):
        # x: torch.Size([2, 25, 32, 64, 64])
        B,an2,C,H,W = x.shape
        x = x.view(-1,C,H,W)
        x = self.conv(x).view(B,an2,C,H//2,W//2)
        x = self.ProgressiveFusion_Block(x)
        return x

class ProgressiveFusion_Block(nn.Module): 
    def __init__(self, nf, ang):
        super(ProgressiveFusion_Block, self).__init__()
        self.ang = ang
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.fusion = nn.Conv2d(nf*self.ang, nf, 1, 1, 0)
        self.conv_decoder = nn.Sequential(nn.Conv2d(2*nf, nf, 3, 1, 1),
                                          nn.LeakyReLU(0.1,inplace=True))
        initialize_weights([self.conv_decoder,self.conv_decoder,self.fusion],0.1)

    def forward(self, x):
        # x.shape BA,C,H,W
        x0_in ,x1_in, x2_in, x3_in, x4_in = x[:,0,:,:,:],x[:,1,:,:,:],x[:,2,:,:,:],x[:,3,:,:,:],x[:,4,:,:,:]
        x5_in ,x6_in, x7_in, x8_in, x9_in = x[:,5,:,:,:],x[:,6,:,:,:],x[:,7,:,:,:],x[:,8,:,:,:],x[:,9,:,:,:]
        x10_in ,x11_in, x12_in, x13_in, x14_in = x[:,10,:,:,:],x[:,11,:,:,:],x[:,12,:,:,:],x[:,13,:,:,:],x[:,14,:,:,:]
        x15_in ,x16_in, x17_in, x18_in, x19_in = x[:,15,:,:,:],x[:,16,:,:,:],x[:,17,:,:,:],x[:,18,:,:,:],x[:,19,:,:,:]
        x20_in ,x21_in, x22_in, x23_in, x24_in = x[:,20,:,:,:],x[:,21,:,:,:],x[:,22,:,:,:],x[:,23,:,:,:],x[:,24,:,:,:]

        x0 = self.conv_encoder(x0_in)
        x1 = self.conv_encoder(x1_in)
        x2 = self.conv_encoder(x2_in)
        x3 = self.conv_encoder(x3_in)
        x4 = self.conv_encoder(x4_in)
        x5 = self.conv_encoder(x5_in)
        x6 = self.conv_encoder(x6_in)
        x7 = self.conv_encoder(x7_in)
        x8 = self.conv_encoder(x8_in)
        x9 = self.conv_encoder(x9_in)
        x10 = self.conv_encoder(x10_in)
        x11 = self.conv_encoder(x11_in)
        x12 = self.conv_encoder(x12_in)
        x13 = self.conv_encoder(x13_in)
        x14 = self.conv_encoder(x14_in)
        x15 = self.conv_encoder(x15_in)
        x16 = self.conv_encoder(x16_in)
        x17 = self.conv_encoder(x17_in)
        x18 = self.conv_encoder(x18_in)
        x19 = self.conv_encoder(x19_in)
        x20 = self.conv_encoder(x20_in)
        x21 = self.conv_encoder(x21_in)
        x22 = self.conv_encoder(x22_in)
        x23 = self.conv_encoder(x23_in)
        x24 = self.conv_encoder(x24_in)


        x_fusion = self.fusion(torch.cat([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,\
                                          x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,\
                                          x20,x21,x22,x23,x24],1))

        x0 = self.conv_decoder(torch.cat([x0, x_fusion], 1))+x0_in
        x1 = self.conv_decoder(torch.cat([x1, x_fusion], 1))+x1_in
        x2 = self.conv_decoder(torch.cat([x2, x_fusion], 1))+x2_in
        x3 = self.conv_decoder(torch.cat([x3, x_fusion], 1))+x3_in
        x4 = self.conv_decoder(torch.cat([x4, x_fusion], 1))+x4_in
        x5 = self.conv_decoder(torch.cat([x5, x_fusion], 1))+x5_in
        x6 = self.conv_decoder(torch.cat([x6, x_fusion], 1))+x6_in
        x7 = self.conv_decoder(torch.cat([x7, x_fusion], 1))+x7_in
        x8 = self.conv_decoder(torch.cat([x8, x_fusion], 1))+x8_in
        x9 = self.conv_decoder(torch.cat([x9, x_fusion], 1))+x9_in
        x10 = self.conv_decoder(torch.cat([x10, x_fusion], 1))+x10_in
        x11 = self.conv_decoder(torch.cat([x11, x_fusion], 1))+x11_in
        x12 = self.conv_decoder(torch.cat([x12, x_fusion], 1))+x12_in
        x13 = self.conv_decoder(torch.cat([x13, x_fusion], 1))+x13_in
        x14 = self.conv_decoder(torch.cat([x14, x_fusion], 1))+x14_in
        x15 = self.conv_decoder(torch.cat([x15, x_fusion], 1))+x15_in
        x16 = self.conv_decoder(torch.cat([x16, x_fusion], 1))+x16_in
        x17 = self.conv_decoder(torch.cat([x17, x_fusion], 1))+x17_in
        x18 = self.conv_decoder(torch.cat([x18, x_fusion], 1))+x18_in
        x19 = self.conv_decoder(torch.cat([x19, x_fusion], 1))+x19_in
        x20 = self.conv_decoder(torch.cat([x20, x_fusion], 1))+x20_in
        x21 = self.conv_decoder(torch.cat([x21, x_fusion], 1))+x21_in
        x22 = self.conv_decoder(torch.cat([x22, x_fusion], 1))+x22_in
        x23 = self.conv_decoder(torch.cat([x23, x_fusion], 1))+x23_in
        x24 = self.conv_decoder(torch.cat([x24, x_fusion], 1))+x24_in

        x_out = torch.cat([x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1),\
                           x5.unsqueeze(1),x6.unsqueeze(1),x7.unsqueeze(1),x8.unsqueeze(1),x9.unsqueeze(1),\
                           x10.unsqueeze(1),x11.unsqueeze(1),x12.unsqueeze(1),x13.unsqueeze(1),x14.unsqueeze(1),\
                           x15.unsqueeze(1),x16.unsqueeze(1),x17.unsqueeze(1),x18.unsqueeze(1),x19.unsqueeze(1),\
                           x20.unsqueeze(1),x21.unsqueeze(1),x22.unsqueeze(1),x23.unsqueeze(1),x24.unsqueeze(1)],1)

        return x_out



if __name__ == "__main__":
    net = Net().cuda()
    input = torch.randn(2, 25, 64, 64).cuda()
    output = net(input)
    print(output.shape)
