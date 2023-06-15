# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import torch.nn as nn
import torch


class model1(nn.Module):

    def __init__(self, level, instance_norm=True, instance_norm_level_1=False):
        super(model1, self).__init__()
        self.level = level
        
        #self.conv1_l1 = ConvMultiBlock(4, 16, 3, instance_norm=False)
        self.conv1_l1 = ConvLayer(4, 16, kernel_size = 3, stride = 1, relu=True, instance_norm =True)
        self.conv1_l2 = ConvLayer(16, 32, kernel_size = 3, stride = 2, relu=True, instance_norm =True)
        self.conv1_l3 = ConvLayer(32, 64, kernel_size = 3, stride = 2, relu=True, instance_norm =True)
        
        ########################  level-3    #####################
       
        self.conv2_l3 = depthwise_conv(64,3) 
        self.cam1 = ChannelAttention(in_channels=64, ratio=1)
        self.sam1 = SpatialAttention(64, 64,kernel_size = 3, stride = 1,dilation = 1)
        self.conv3_l3 = ConvLayer(64, 32, kernel_size = 5, stride = 1, relu=True, instance_norm =True)
        self.conv4_l3 = depthwise_conv(32,5)  
        self.conv5_l3 = ConvLayer(32, 64, kernel_size = 3, stride = 1, relu=True, instance_norm =True)
        self.conv6_l3 = ConvLayer(64, 32, kernel_size = 1, stride = 1, relu=True, instance_norm =True)
        self.upsample3 = UpsampleConvLayer(64, 32, 3) 
        self.conv_l3_out = ConvLayer(64, 3, kernel_size = 3, stride = 1, relu=False, instance_norm =True)
        self.act3 = nn.Tanh()
        
        ####################### level-2 ########################3 
       
        self.conv2_l2 = ConvLayer(64, 128, kernel_size = 3, stride = 1, relu=True, instance_norm =True)
        self.conv3_l2 = depthwise_conv(64,3)
        self.conv4_l2 = ConvLayer(64, 128, kernel_size = 3, stride = 1, relu=True, instance_norm =True)        
        self.conv5_l2 = ConvLayer(256, 64, kernel_size = 1, stride = 1, relu=True, instance_norm =True)
        self.cam2 = ChannelAttention(in_channels=64, ratio=1)
        self.sam2 = SpatialAttention(64, 64,kernel_size = 3, stride = 1,dilation = 1)
        self.upsample2 = UpsampleConvLayer(64, 32, 3)
        self.conv_l2_out = ConvLayer(64, 3, kernel_size = 3, stride = 1, relu=False, instance_norm =True)
        self.act2 = nn.Tanh()
        ################# level-1 #######################
        
        self.conv_IEM_start = ConvLayer(32, 4, kernel_size = 1, stride = 1, relu=True)
        self.conv1_IEM= ConvLayer(4, 16, kernel_size = 5, stride = 1, relu=True)
        self.conv2_IEM= ConvLayer(4, 16, kernel_size = 7, stride = 1, relu=True)
        self.conv3_IEM= ConvLayer(4, 16, kernel_size = 9, stride = 1, relu=True)
        self.IEM1 = IEM_module(in_channels=16)
        self.IEM2 = IEM_module(in_channels=16)
        self.IEM3 = IEM_module(in_channels=16)
        self.IEM4 = IEM_module(in_channels=16)
        self.conv2_l1 = ConvLayer(64, 256, kernel_size = 3, stride = 1, relu=True) 
        self.conv3_l1 = ConvLayer(256, 256, kernel_size = 1, stride = 1, relu=True)
        self.pix_shuff2 = nn.PixelShuffle(2)
        self.out_att1 = att_module(input_channels=32, ratio=2, kernel_size=3)
        self.conv4_l1 = ConvLayer(32, 96, kernel_size = 3, stride = 1, relu=True)
        self.conv5_l1 = ConvLayer(96, 256, kernel_size = 3, stride = 1, relu=True) 
        self.pix_shuff1 = nn.PixelShuffle(2)
        self.conv6_l1 = ConvLayer(64, 3, 3, stride=1,relu = False)
        self.act1 = nn.Tanh()
        
    def level_3(self, conv1_l3):
        z1_l3 = self.conv2_l3(conv1_l3)
        #print('z1_l3 shape: ',z1_l3.shape)
        cam1 = self.cam1(z1_l3)
        #print('cam1 shape: ',cam1.shape)
        z2_l3 = z1_l3 * cam1
        #print('z2_l3 shape: ',z2_l3.shape)
        sam1 = self.sam1(z2_l3)
        #print('sam1 shape: ',sam1.shape)
        z3_l3 = z2_l3 * sam1
        #print('z3_l3 shape: ',z3_l3.shape)
        z4_l3 = self.conv3_l3(z3_l3)
        #print('z4_l3 shape: ',z4_l3.shape)
        z5_l3 = self.conv4_l3(z4_l3)
        #print('z5_l3 shape: ',z5_l3.shape)
        z6_l3 = self.conv5_l3 (z5_l3)
        #print('z6_l3 shape: ',z6_l3.shape)
        z7_l3 = self.conv6_l3 (z6_l3)
        #print('z7_l3 shape: ',z7_l3.shape)
        z8_l3 = torch.cat([z4_l3, z7_l3], 1)
        #print('z8_l3 shape: ',z8_l3.shape)
        l3_upsample = self.upsample3(z8_l3)
        #print('l3_upsample shape: ',l3_upsample.shape)
        l3_out = self.act3(self.conv_l3_out (z8_l3))
        #print('l3_out shape: ',l3_out.shape)
        return l3_out, l3_upsample
  
    def level_2(self, conv1_l2, l3_upsample):
                
        z1_l2 = torch.cat([conv1_l2, l3_upsample], 1)
        #print('z1_l2 shape: ',z1_l2.shape)
        z2_l2 = self.conv2_l2(z1_l2)
        #print('z2_l2 shape: ',z2_l2.shape)
        z3_l2 = self.conv3_l2(z1_l2)
        #print('z3_l2 shape: ',z3_l2.shape)
        z4_l2 = self.conv4_l2(z3_l2)
        #print('z4_l2 shape: ',z4_l2.shape)
        z5_l2 = torch.cat([z2_l2, z4_l2], 1)
        #print('z5_l2 shape: ',z5_l2.shape) 
        z6_l2 = self.conv5_l2(z5_l2)
        #print('z6_l2 shape: ',z6_l2.shape) 
        sam2 = self.sam2(z1_l2)
        #print('sam2 shape: ',sam2.shape)
        cam2 = self.cam2(sam2)
        #print('cam2 shape: ',cam2.shape)
        z7_l2 = cam2 + z6_l2
        #print('z7_l2 shape: ',z7_l2.shape) 
         
        l2_upsample = self.upsample2(z7_l2)
        #print('l2_upsample shape: ',l2_upsample.shape)
        l2_out = self.act2(self.conv_l2_out (z7_l2))
        #print('l2_out shape: ',l2_out.shape)
        return l2_out, l2_upsample
    
    def level_1(self, conv1_l1_, l2_upsample):
        
        #print('conv1_l1 shape: ', conv1_l1_.shape)
        #print('l2_upsample',l2_upsample.shape)
        IEM_start = self.conv_IEM_start(l2_upsample)
        #print('l2_upsample',l2_upsample.shape)
        a1 = self.conv1_IEM(IEM_start)
        #print('conv1 shape: ',a1.shape)
        IEM_a1 = self.IEM1(a1)
        #print('IEM1 shape: ',IEM_a1.shape)
        a2 = self.conv2_IEM(IEM_start)
        #print('conv2 shape: ',a2.shape)
        IEM_a2 = self.IEM2(a2)
        #print('IEM2 shape: ',IEM_a2.shape)
        a3 = self.conv3_IEM(IEM_start)
        #print('conv3 shape: ',a3.shape)
        IEM_a3 = self.IEM3(a3)
        #print('IEM3 shape: ',IEM_a3.shape)
        IEM_concat = torch.cat([conv1_l1_, IEM_a1, IEM_a2, IEM_a3], dim=1)
        #print('IEM_concat shape: ',IEM_concat.shape)
        z1_l1 = self.conv2_l1(IEM_concat)
        #print('z1_l1 shape: ',z1_l1.shape)
        z2_l1=self.conv3_l1(z1_l1)
        #print('z2_l1 shape: ',z2_l1.shape) 
        pixel_shuffle_2 = self.pix_shuff2(z2_l1)

        att1 = self.out_att1(l2_upsample)
        #print('att1 shape: ',att1.shape)
        z3_l1 = att1 + l2_upsample
        #print('z3_l1 shape: ',z3_l1.shape)
        z4_l1 = self.conv4_l1(z3_l1)
        #print('z4_l1 shape: ',z4_l1.shape)
        z5_l1 = self.conv5_l1(z4_l1)
        #print('z5_l1 shape: ',z5_l1.shape)
        pixel_shuffle_1 = self.pix_shuff1(z5_l1)
        #print('pixel_shuffle shape: ',pixel_shuffle_1.shape) ### 16
        
        z6_l1 = pixel_shuffle_2 * pixel_shuffle_1
        #print('z6_l1 shape: ',z6_l1.shape)

        out = self.act1(self.conv6_l1(z6_l1))
        #print('out shape: ',out.shape)
        return out
              
    def forward(self, x):
        
       conv1_l1_ = self.conv1_l1(x)     
       conv1_l2_ = self.conv1_l2(conv1_l1_)
       conv1_l3_ = self.conv1_l3(conv1_l2_)
       
       l3_out, l3_upsample =self.level_3(conv1_l3_)

       if self.level < 3:
           l2_out, l2_upsample = self.level_2(conv1_l2_, l3_upsample)
       if self.level < 2:
           out = self.level_1(conv1_l1_, l2_upsample)
       
       if self.level == 1:
           enhanced = out
       if self.level == 2:
           enhanced = l2_out
       if self.level == 3:
           enhanced = l3_out
       
       return enhanced

class ConvMultiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, instance_norm):

        super(ConvMultiBlock, self).__init__()
        
        self.conv_a = ConvLayer(in_channels, out_channels, kernel_size, stride=1, instance_norm=instance_norm)
        self.conv_b = ConvLayer(out_channels, out_channels, kernel_size, stride=1, instance_norm=instance_norm)
        
    def forward(self, x):

        out = self.conv_a(x)
        output_tensor = self.conv_b(out)
        return output_tensor

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True, instance_norm=False):

        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2

        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.instance_norm = instance_norm
        self.instance = None
        self.relu = None

        if instance_norm:
            self.instance = nn.InstanceNorm2d(out_channels, affine=True)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.instance_norm:
            out = self.instance(out)

        if self.relu:
            out = self.relu(out)

        return out
    
class depthwise_conv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(depthwise_conv, self).__init__()
        
        #print('kernel_size: ',kernel_size)
        reflection_padding = 2*(kernel_size//2)
        
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.dw_conv =  nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, dilation=2, groups=in_channels),nn.ReLU())
        self.point_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        
        y = self.reflection_pad(x)
        #print('y_depthwise shape: ',y.shape)
        conv1 = self.dw_conv(y)
        #print('depthwise_conv shape: ',conv1.shape)
        conv2 = self.point_conv(conv1)
        #print('point_conv shape: ',conv2.shape)
        out = self.sigmoid(conv2)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, dilation):
        
        super(SpatialAttention, self).__init__()
        #reflection_padding = kernel_size//2
        #self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        
        self.dw_conv =  nn.Conv2d(input_channels, output_channels, kernel_size, stride, dilation, groups= input_channels)
        self.relu = nn.ReLU()
        self.point_conv = nn.Conv2d(output_channels, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #print('x shape: ',x.shape)
        #y = self.reflection_pad(x)
        #print('sa refl_pad shape: ',y.shape)
        conv1 = self.dw_conv(x)
        #print('depthwise_conv shape: ',conv1.shape)
        act_relu =self.relu(conv1) 
        
        conv2 = self.point_conv(act_relu)
        #print('point_conv shape: ',conv2.shape)
        act_conv2 = self.sigmoid(conv2)
        out = x * act_conv2
        #print('out_sa shape: ',out.shape)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, kernel_size = 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_channels // ratio, in_channels, kernel_size= 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        #print('x_ca_avg shape: ',avg_out.shape)
        max_out = self.fc(self.max_pool(x))
        #print('x_ca_max shape: ',max_out.shape)
        out = self.sigmoid(avg_out * max_out)
        #print('ca_out1 shape: ',out.shape)
        out_ca = out * x
        #print('ca_out shape: ',out_ca.shape)
        return out_ca

class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, upsample=2, stride=1, relu=True):

        super(UpsampleConvLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample, mode='bilinear', align_corners=True)

        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)

        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        if relu:
            self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        out = self.upsample(x)
        out = self.reflection_pad(out)
        out = self.conv2d(out)

        if self.relu:
            out = self.relu(out)

        return out

class att_module(nn.Module):
    
    def __init__(self, input_channels, ratio, kernel_size, instance_norm=False):
        super(att_module, self).__init__()
        
        self.conv1 = ConvLayer(in_channels= input_channels, out_channels=input_channels*2, kernel_size=3, stride=1, relu=True)
        self.conv2 = ConvLayer(in_channels=input_channels*2, out_channels=input_channels*2, kernel_size=1, relu=True, stride =1)
        
        self.ca = ChannelAttention(input_channels*2, ratio)
        #self.sa = SpatialAttention(in_channels, kernel_size=5, dilation=2)
        self.sa = SpatialAttention(input_channels*2, input_channels*2, kernel_size=5, stride = 1, dilation=2)
        self.conv3 = ConvLayer(input_channels*4, input_channels, kernel_size=1, stride= 1, relu=True)
    
    def forward(self, x):
       
       conv1 = self.conv1(x)
       #print('conv1_att shape: ',conv1.shape)
       conv2 = self.conv2(conv1)
       #print('conv2_att shape: ',conv2.shape)
              
       z1 = self.ca(conv2)
       #print('z1_att shape: ',z1.shape)
       z2 = self.sa(conv2)
       #print('z2_att shape: ',z2.shape)
       out = self.conv3(torch.cat([z1, z2], 1))
       #print('out_att shape: ',out.shape)
       return out
   
class IEM_module(nn.Module):
    def __init__(self, in_channels):
    
        super(IEM_module, self).__init__()
        #print('in channels : ', in_channels)
        #reflection_padding = 3//2
        #self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv1 = ConvLayer(in_channels, 16, 3, 1, relu=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv2 = ConvLayer(in_channels, 8, 1, 1, relu= True)
        self.conv3 = ConvLayer(8, 8, 1, 1, relu= True)
        self.conv4 = ConvLayer(8, 16, 1, 1, relu= True)
        
    def forward(self, x):
       #print('x_att shape: ',x.shape)
       #ref_pad = self.reflection_pad(x)
       #print('ref_pad_att shape: ',ref_pad.shape)
       z1 = self.conv1(x)
       #print('z1 shape: ',z1.shape)
       
       avg_pool_out = self.avg_pool(x)
       #print('avg_pool shape: ',avg_pool_out.shape)
       z2 = self.conv2(avg_pool_out)
       #print('conv2 shape: ',z1.shape)
       z3 = self.conv3(z2)
       #print('conv3 shape: ',z3.shape)
       z4 = self.conv4(z3)
       #print('conv4 shape: ',z3.shape) 
       
       out = z1 * z4 * x
       #print('out shape: ',out.shape)
       
       return out
