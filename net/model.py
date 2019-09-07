import torch
import torch.nn as nn
import torch.nn.functional as F
from net.net import ConvLayer as ConvLayer
from net.net import UpsampleConvLayer as  UpsampleConvLayer
from net.net import BasicBlock_Residual as BasicBlock_Residual

class net_residual(nn.Module):
    def __init__(self, n=1):
        super(net_residual, self).__init__()
        #256*256
        self.n = n
        self.pading = nn.ReflectionPad2d(2)
        self.conv2_init_1 = nn.Conv2d(3, 8*self.n, kernel_size=5)
        self.conv2_init_2 = nn.Conv2d(8*self.n, 16*self.n, kernel_size=5)

        self.conv2_1_1 = BasicBlock_Residual(16*self.n, 16*self.n, 1)
        self.conv2_1_2 = ConvLayer(16*self.n, 32*self.n, kernel_size=3, stride=2)    #  b * 32 * h/2 * w/2   b*32*128*128
        self.conv2_2_1 = BasicBlock_Residual(32*self.n, 32*self.n, 1)
        self.conv2_2_2 = ConvLayer(32*self.n, 64*self.n, kernel_size=3, stride=2)    #  b * 64 * h/2 * w/2
        self.conv2_3_1 = BasicBlock_Residual(64*self.n, 64*self.n, 1)                # 2+1 --> 3   +:cat
        self.conv2_3_2 = ConvLayer(64*self.n, 128*self.n, kernel_size=3, stride=2)   #  b * 128 * h/2 * w/2
        self.conv2_4_1 = BasicBlock_Residual(128*self.n, 128*self.n, 1)                # 3+2 --> 4   +:cat

        self.conv2_4_2 = ConvLayer(128*self.n, 256*self.n, kernel_size=3, stride=2)  #  b * 256 * h/2 * w/2
        self.conv2_5_1 = BasicBlock_Residual(256*self.n, 256*self.n, 1)
        self.conv2_5_2 = BasicBlock_Residual(256*self.n, 256*self.n, 1)
        self.conv2_5_3 = BasicBlock_Residual(256*self.n, 256*self.n, 1)
        # self.conv2_5_2 = ConvLayer(256, 512, kernel_size=3, stride=2)
        # # 待定...
        # self.conv2b_5_1 = UpsampleConvLayer(512, 256, kernel_size=3, stride=2)
        self.conv2b_5_2 = BasicBlock_Residual(256*self.n, 256*self.n)
        self.conv2b_4_1 = UpsampleConvLayer(256*self.n, 128*self.n, kernel_size=3, stride=2)       # 长宽比对应unet的长宽小一    (h-1)*s - 2*p + k
        self.conv2b_4_2 = BasicBlock_Residual(128*self.n, 128*self.n)                           # 2+b_4_1 --> b_4_2
        self.conv2b_3_1 = UpsampleConvLayer(128*self.n, 64*self.n, kernel_size=3, stride=2)
        self.conv2b_3_2 = BasicBlock_Residual(64*self.n, 64*self.n)                                # 1+b_3_1 --> b_3_2
        self.conv2b_2_1 = UpsampleConvLayer(64*self.n, 32*self.n, kernel_size=3, stride=2)
        self.conv2b_2_2 = BasicBlock_Residual(32*self.n, 32*self.n)                            # 2+4b --> 2b
        self.conv2b_1_1 = UpsampleConvLayer(32*self.n, 16*self.n, kernel_size=3, stride=2)
        self.conv2b_1_2 = BasicBlock_Residual(16*self.n,16*self.n)                              # 1+3b --> 1b

        self.conv_finish = ConvLayer(16*self.n, 3, kernel_size=3, stride=1)

        # self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=4)
        # self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.down_1_3 = nn.Conv2d(16*self.n, 64*self.n, kernel_size=3, stride=4)
        self.down_2_4 = nn.Conv2d(32*self.n, 128*self.n, kernel_size=3, stride=4)
        self.down_3_5 = nn.Conv2d(64*self.n, 256*self.n, kernel_size=3, stride=4)
		
		if self.n == 1:
			self.relu = nn.LeakyReLU(0.2)
		if self.n == 2:
			self.relu = nn.ReLU()

        self.upb_4_2 = UpsampleConvLayer(128*self.n, 32*self.n, kernel_size=3, stride=4)
        self.upb_3_1 = UpsampleConvLayer(64*self.n, 16*self.n, kernel_size=3, stride=4)
        self.upb_5_3 = UpsampleConvLayer(256*self.n, 64*self.n, kernel_size=3, stride=4)

    def forward(self,x):

        out = self.pading(x)
        out = self.relu(self.conv2_init_1(out))
        out = self.pading(out)
        out = self.relu(self.conv2_init_2(out))
        # 1
        out_1 = self.conv2_1_1(out)  # 256*256
        out = self.relu(self.conv2_1_2(out_1))  # 16-->32  128*128
        #2
        out_2 = self.conv2_2_1(out)  # 32 128*128
        out = self.relu(self.conv2_2_2(out_2))  # 32-->64  64*64
        #3
        out_3_1 = self.relu(self.down_1_3(out_1))  # 1->3  16-->64 64*64
        out = F.interpolate(out, out_3_1.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_3_1)  # 1+2-->3
        # del out_3_1
        out_3 = self.conv2_3_1(out)  # 64 64*64
        out = self.relu(self.conv2_3_2(out_3))  # 64-->128 32*32
        #4
        out_4_1 = self.relu(self.down_2_4(out_2)) # 2->4  128 32*32
        out = F.interpolate(out, out_4_1.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_4_1)  # 2+3-->4
        out_4 = self.conv2_4_1(out)      # 128 32*32
        out = self.relu(self.conv2_4_2(out_4)) # 128-->256
        #5
        out_5_1 = self.relu(self.down_3_5(out_3)) # 3->5 256
        out = F.interpolate(out, out_5_1.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_5_1)
        out = self.conv2_5_1(out)
        out = self.conv2_5_2(out)
        out = self.conv2_5_3(out)

        #b 5
        out = F.interpolate(out, out_5_1.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_5_1)
        out_b_5 = self.conv2b_5_2(out)
        #b_4
        out = self.relu(self.conv2b_4_1(out_b_5))
        out = F.interpolate(out, out_4_1.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_4_1)  # 2+4-->b_4  32*32
        out = F.interpolate(out, out_4.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_4)
        out_b_4 = self.conv2b_4_2(out)  # 128 --> 128  32*32
        #b_3
        out = self.relu(self.conv2b_3_1(out_b_4))  # 128 --> 64  63*63     k=2 p=0 -->64*64
        out = F.interpolate(out, out_3.size()[2:], mode='bilinear', align_corners=True)  # 64 64*64
        out = torch.add(out, out_3)     # 64 64*64
        out = F.interpolate(out, out_3_1.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_3_1)   # 1+3-->b_3 64 64*64
        out_b_5 = self.relu(self.upb_5_3(out_b_5))
        out_b_5 = F.interpolate(out_b_5, out.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_b_5)
        out_b_3 = self.conv2b_3_2(out)  # 64 --> 64  64*64
        #b_2
        out = self.relu(self.conv2b_2_1(out_b_3))  #64-->32  127*127
        out = F.interpolate(out, out_2.size()[2:], mode='bilinear', align_corners=True)  # 32  128*128
        out = torch.add(out, out_2)  # 32  128*128
        out_b_4 = self.relu(self.upb_4_2(out_b_4)) # 128 --> 32 125*125  if k=4 p=0 128*128
        out_b_4 = F.interpolate(out_b_4, out.size()[2:], mode='bilinear', align_corners=True)
        out = torch.add(out, out_b_4)  # b_4 + b_3 --> b_2
        out = self.conv2b_2_2(out)  # 32 128*128
        #b_1
        out = self.relu(self.conv2b_1_1(out))  #32-->16 255*255
        out = F.interpolate(out, out_1.size()[2:], mode='bilinear', align_corners=True)  # 16 256*256
        out = torch.add(out, out_1)  # 16 256*256
        out_b_3 = self.relu(self.upb_3_1(out_b_3)) # 16 253*253
        out_b_3 = F.interpolate(out_b_3, out.size()[2:], mode='bilinear', align_corners=True)  # 16 256*256
        out = torch.add(out, out_b_3)  # b_3 + b_2 --> b_1 16 256*256
        out = self.conv2b_1_2(out) # 16 256*256

        out = self.conv_finish(out) # 3 256*256

        return out
