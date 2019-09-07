import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation= dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class BasicBlock_Residual_resnet(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Residual_resnet, self).__init__()
        self.pading = nn.ReflectionPad2d(1)
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        # identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += x

        return self.relu(out)

class BasicBlock_Residual2(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Residual2, self).__init__()
        self.pading = nn.ReflectionPad2d(1)
        self.conv_init = ConvLayer(in_planes, in_planes, kernel_size=1, stride = stride)
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride = stride)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride = stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        residul = self.conv_init(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = torch.add(out,residul)
        return out

        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = torch.add(out,residul)
        return out
