import torch
import torch.nn as nn
import torch.nn.functional as F
from .deconv import FastDeconv
from .deform import DeformConv2d
import numpy as np
import time



class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.BatchNorm2d(in_features),
                        nn.PReLU(),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.BatchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class ChannelAttention(nn.Module):
    def __init__(self, nc,number, norm_layer = nn.BatchNorm2d):
        super(ChannelAttention, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn1 = norm_layer(nc)
        self.prelu = nn.PReLU(nc)
        self.conv2 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn2 = norm_layer(nc)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(nc, number, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        se = self.gap(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return se
    
class SpatialAttention(nn.Module):
    def __init__(self, nc, number, norm_layer = nn.BatchNorm2d):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = norm_layer(nc)
        self.prelu = nn.PReLU(nc)
        self.conv2 = nn.Conv2d(nc,number,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = norm_layer(number)
        
        self.conv3 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=3,dilation=3,bias=False)
        self.conv4 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=5,dilation=5,bias=False)
        self.conv5 = nn.Conv2d(number,number,kernel_size=3,stride=1,padding=7,dilation=7,bias=False)
        
        self.fc1 = nn.Conv2d(number*4,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        x1 = x
        x2 = self.conv3(x)
        x3 = self.conv4(x)
        x4 = self.conv5(x)
        
        se = torch.cat([x1, x2, x3, x4], dim=1)
        
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        
        return se

class _residual_block_ca(nn.Module):
    def __init__(self,nc, number = 4, norm_layer = nn.BatchNorm2d):
        super(_residual_block_ca,self).__init__()
        self.CA = ChannelAttention(nc,number)
        self.MSSA = SpatialAttention(nc,number)
    def forward(self,x):
        x0 = x
        x1 = self.CA(x)*x
        x2 = self.MSSA(x1)*x1
        
        return x0+x2

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, in_features=32, n_residual_att=6, n_residual_blocks=6):
        super(Generator, self).__init__()
        # 预处理
        self.deconv = FastDeconv(input_nc,input_nc,3,padding=1)
	
	    # 注意力
        att = [ nn.ReflectionPad2d(3),
	            nn.Conv2d(input_nc, in_features//2, 7),
                nn.BatchNorm2d(in_features//2),
                nn.PReLU()]
        for _ in range(n_residual_att):
            att += [_residual_block_ca(in_features//2)]
    
        att += [nn.ReflectionPad2d(3),
                nn.Conv2d(in_features//2, 1, 7),
                nn.Sigmoid()]
    
        self.att = nn.Sequential(*att)

        # 主干第一层
        conv1 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, in_features, 7),
                    nn.BatchNorm2d(in_features),
                    nn.PReLU()]
        for _ in range(3):
            conv1 += [  nn.Conv2d(in_features, in_features, 3,padding=1),
                        nn.BatchNorm2d(in_features),
                        nn.PReLU() ]
        self.conv1  = nn.Sequential(*conv1)
    
    

        # 主干U-Net残差层       
        model = []

        # 下采样
        in_features = in_features
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.BatchNorm2d(out_features),
                        nn.PReLU() ]
            in_features = out_features
            out_features = in_features*2

        # 残差模块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        for _ in range(2):
            model += [  DeformConv2d(in_features,in_features), 
                        nn.BatchNorm2d(in_features),
                        nn.PReLU() ]
        # 上采样
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(out_features),
                        nn.PReLU() ]
            in_features = out_features
            out_features = in_features//2

        # 输出
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Tanh() ]
        
        
        self.model = nn.Sequential(*model)
        
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.25)
        self.gamma_max = 0.1
        self.gamma_min = 0.01

    def forward(self, x, m_gt=None, l_mask=0.00001):
        beta = 1.0
        if l_mask >= self.gamma_max:
            beta = 1.0
        elif l_mask >= self.gamma_min:
            beta = (l_mask - self.gamma_min)/(self.gamma_max - self.gamma_min)
        else:
            beta = 0.0
        
        x_deconv = self.deconv(x)
        
        m_g = self.att(x_deconv)
        
        if m_gt is None:
            m = m_g
        else:
            m = beta*m_gt + (1-beta)*m_g
        
        x_in  = self.conv1(x_deconv)
        
        x_inp = self.alpha * m * x_in + (1-self.alpha) * x_in
        
        x_out = self.model(x_inp)
        return x_out, m_g



class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.BatchNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.BatchNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1))
        
if __name__ == '__main__':
    net = Generator(3,3).cuda()
    input_tensor = torch.Tensor(np.random.random((1,3,1500,1000))).cuda()
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(input_tensor.shape) 

