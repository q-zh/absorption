import torch.nn as nn
import torch
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class HNet(nn.Module):
    """HNet network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(HNet, self).__init__()

        layers = []
        layers.append(nn.Conv2d(4, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU())

        # 96*144*64
        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            curr_dim = curr_dim * 2
        # 24*36*256
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        # 24*36*256
        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            # layers.append(nn.Upsample(scale_factor=4, mode='nearest'))
            # layers.append(nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.ReLU6())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # x = x*0.5+0.5
        c = c.view(c.size(0), 1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        
        return self.main(x)/6

class GNet(nn.Module):
    """GNet network."""
    def __init__(self, image_size=256, conv_dim=64, c_dim=1, repeat_num=5):
        super(GNet, self).__init__() 
        layers1 = []
        # 96*144
        layers1.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers1.append(nn.ReLU6())
        curr_dim = conv_dim
        # 48*72 * 64
        layers2 = []
        layers2.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers2.append(nn.ReLU6())
        curr_dim = curr_dim * 2
        # 48*72 * 64
        layers3 = []
        layers3.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers3.append(nn.ReLU6(inplace=True))
        curr_dim = curr_dim * 2
        # 48*72 * 64
        layers4 = []
        layers4.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers4.append(nn.ReLU6(inplace=True))
        curr_dim = curr_dim * 2
        # 48*72 * 64
        layers5 = []
        layers5.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers5.append(nn.ReLU6())
        curr_dim = curr_dim * 2
      
        layers6 = []
        layers6.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=1, padding=1, bias=False))
        layers6.append(nn.AvgPool2d(kernel_size=4, stride=2, padding=1))
        # layers6.append(nn.Dropout2d(p=0.2))
        layers6.append(nn.Sigmoid())
        # layers.append(nn.Conv2d(curr_dim*2, curr_dim*4, kernel_size=2, stride=1, padding=1))
        
        self.BCE = nn.BCELoss()
        self.L1LOSS = nn.L1Loss()

        self.main1 = nn.Sequential(*layers1)
        self.main2 = nn.Sequential(*layers2)
        self.main3 = nn.Sequential(*layers3)
        self.main4 = nn.Sequential(*layers4)
        self.main5 = nn.Sequential(*layers5)
        self.main6 = nn.Sequential(*layers6)

        
        self.linear = nn.Linear(curr_dim*8, 1, bias=False)
        self.sg = nn.Sigmoid()
    
    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def compute_loss(self, x1, y1, a):
        x1 = x1.view(x1.size(0), -1)
        y1 = y1.view(y1.size(0), -1)
        loss1 = torch.div(x1+1e-10, y1+1e-10).clamp_(min=0.7, max=1.0)
        loss1_mask = (loss1<1)&(loss1>0.7)
        # print(loss1_mask.mean(1))
        loss1_mask = loss1_mask.float()
        tensor_a = torch.ones(loss1.size()).cuda()*a
        loss1_error1 = self.BCE((loss1-0.7)/0.3, (tensor_a-0.7)/0.3)
        # yn = torch.mul(y1, a)
        # loss = self.L1LOSS(yn, x1)
        loss1_tensor = torch.mul(loss1_error1, loss1_mask)
        loss1_out = torch.div(torch.sum(loss1_tensor, 1)+1e-10, torch.sum(loss1_mask, 1)+1e-10).unsqueeze(1)
       
        return loss1_out, torch.abs(torch.mul(x1, loss1_mask)).sum()

    def forward(self, x, y, a):
        # x = x*0.5+0.5
        # y = y*0.5+0.5
        x1 = self.main1(x)
        mu = torch.mean(torch.mean(x1,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        x1 = x1 - mu
        # x1 = x1.clamp_(min=0.0, max=1.0)
        x2 = self.main2(x1)
        mu = torch.mean(torch.mean(x2,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        x2 = x2 - mu
        # x2 = x2.clamp_(min=0.0, max=1.0)
        x3 = self.main3(x2)
        mu = torch.mean(torch.mean(x3,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        x3 = x3  - mu
        # x3 = x3.clamp_(min=0.0, max=1.0)
        x4 = self.main4(x3)
        mu = torch.mean(torch.mean(x4,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        x4 = x4 - mu
        # x4 = x4.clamp_(min=0.0, max=1.0)
        x5 = self.main5(x4)
        mu = torch.mean(torch.mean(x5,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        x5 = x5 - mu
        # x5 = x5.clamp_(min=0.0, max=1.0)
        x6 = self.main6(x5)

        y1 = self.main1(y)
        mu = torch.mean(torch.mean(y1,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        y1 = y1 - mu
        # y1 = y1.clamp_(min=0.0, max=1.0)
        y2 = self.main2(y1)
        mu = torch.mean(torch.mean(y2,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        y2 = y2 - mu
        # y2 = y2.clamp_(min=0.0, max=1.0)
        y3 = self.main3(y2)
        mu = torch.mean(torch.mean(y3,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        y3 = y3 - mu
        # y3 = y3.clamp_(min=0.0, max=1.0)
        y4 = self.main4(y3)
        mu = torch.mean(torch.mean(y4,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        y4 = y4 - mu
        # y4 = y4.clamp_(min=0.0, max=1.0)
        y5 = self.main5(y4)
        mu = torch.mean(torch.mean(y5,dim=2),dim=2).unsqueeze(-1).unsqueeze(-1)
        y5 = y5 - mu
        # y5 = y5.clamp_(min=0.0, max=1.0)
        y6 = self.main6(y5)

        loss1, _ = self.compute_loss(x1, y1, a)
        loss2, _ = self.compute_loss(x2, y2, a)
        loss3, _ = self.compute_loss(x3, y3, a)
        loss4, _ = self.compute_loss(x4, y4, a)
        loss5, loss5_out = self.compute_loss(x5, y5, a)

        loss = 0.02*loss1 + 0.08*loss2 + 0.2*loss3 + 0.3*loss4 + 0.4*loss5

        x7 = x6.view(-1, self.num_flat_features(x6))
        #print(h.size())
        #print(g.size())
        x8 = self.sg(self.linear(x7))*0.3+0.7

        y7 = y6.view(-1, self.num_flat_features(y6))
        #print(h.size())
        #print(g.size())
        y8 = self.sg(self.linear(y7))*0.3+0.7
            # print(out_cls.shape)
        return x8.view(x6.size(0), -1),  y8.view(x6.size(0), -1), loss, loss5_out