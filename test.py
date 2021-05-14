import os
import argparse
from torch.backends import cudnn
from torchvision import transforms as T
import os
from PIL import Image
from torch.utils import data
import random
import torch

import pytorch_ssim

from model import HNet
from torchvision.utils import save_image
import numpy as np
import torch
import time
import datetime
import os
from vgg16 import Vgg16
# from torch.utils.serialization import load_lua
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
import torchfile
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

    def forward(self, x):
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


        x7 = x6.view(-1, self.num_flat_features(x6))
        #print(h.size())
        #print(g.size())
        x8 = self.sg(self.linear(x7))*0.3+0.7


        return x8.view(x6.size(0), -1)

class Solver(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = GNet().to(self.device)
        self.D = HNet().to(self.device)
        self.ssim_si_loss = pytorch_ssim.SSIM_SI_SIP(size_average=False)
        self.vgg = Vgg16()
        self.init_vgg16('../models/')
        self.vgg.load_state_dict(torch.load(os.path.join('../models/', "vgg16.weight")))
        self.vgg.cuda()
        self.criterionCAE = nn.L1Loss()

    def init_vgg16(self, model_folder):
        """load the vgg16 model feature"""
        if not os.path.exists(os.path.join(model_folder, 'vgg16.weight')):
            if not os.path.exists(os.path.join(model_folder, 'vgg16.t7')):
                os.system(
                    'wget http://cs.stanford.edu/people/jcjohns/fast-neural-style/models/vgg16.t7 -O ' + os.path.join(
                        model_folder, 'vgg16.t7'))
            vgglua = torchfile.load(os.path.join(model_folder, 'vgg16.t7'))
            vgg = Vgg16()
            for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
                dst.data[:] = src
            torch.save(vgg.state_dict(), os.path.join(model_folder, 'vgg16.weight'))

    def get_perceptual_loss(self, target, x_hat):
        # Perceptual Loss 1
        features_content = self.vgg(target)
        features_y = self.vgg(x_hat)
        f_xc_c = Variable(features_content[1].data, requires_grad=False)
        content_loss = self.criterionCAE(features_y[1], f_xc_c)
        f_xc_c = Variable(features_content[0].data, requires_grad=False)
        content_loss1 = self.criterionCAE(features_y[0], f_xc_c)

        return content_loss + content_loss1

    def denorm(self, x):
        out = x #= (x + 1) / 2
        return out#.clamp_(0, 1)

    def get_loss(self, batch1, batch2):
        s1 = self.denorm(batch1)
        s2 = self.denorm(batch2)
        ssim, si, sip = self.ssim_si_loss(s1, s2)
        perceptual_loss = self.get_perceptual_loss(batch1, batch2)
        psnr = ((s1 - s2) ** 2).mean()
        psnr = 10 * (torch.log10(1.0 / psnr))
        loss = - ssim - si - psnr / 40.0 + 3 + 2 * perceptual_loss
        return loss, ssim, perceptual_loss, psnr, sip

    def restore_model(self):
        G_path = os.path.join('./ours/models/G.ckpt')
        D_path = os.path.join('./ours/models/D.ckpt')
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        print('Loading the trained models' + G_path + D_path)

    def compute_mean(self, data_loader, name):
        datafile_dir = os.path.join(config.output_dir, name)
        if not os.path.exists(datafile_dir):
            os.makedirs(datafile_dir)

        f = open(os.path.join(datafile_dir, 'test.txt'), 'a+')
        ssim_list = []
        sip_list = [] 
        psnr_list = []
        with torch.no_grad():
            for i, (mix, fname) in enumerate(data_loader):
                mix= mix.to(self.device)
                fake_out = self.G(mix)
                fake = self.D(mix, fake_out)
                loss, ssim, perceptual_loss, psnr, sip = self.get_loss(gro, fake)
                ssim_list.append(ssim.mean().item())
                sip_list.append(sip.mean().item())
                psnr_list.append(psnr.mean().item())
                f.write(str(''.join(fname)) + '\t' + str(fake_out.item()) + '\t' + str(ssim.mean().item()) + '\t' + str(sip.mean().item()) + '\t' + str(
                    psnr.mean().item()) + '\n')

                print('{} \t {}, {}, {}, {}'.format(str(''.join(fname)), str(fake_out.item()), ssim.item(), sip.item(), psnr.item()))

            
                img_dir = os.path.join(datafile_dir, 'results')
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                img_list = [self.denorm(mix)]
                img_list.append(self.denorm(fake))
                img_concat = torch.cat(img_list, dim=3)
                if not os.path.exists(os.path.join(img_dir, 'out')):
                    os.makedirs(os.path.join(img_dir, 'out'))
                if not os.path.exists(os.path.join(img_dir, 'mst')):
                    os.makedirs(os.path.join(img_dir, 'mst'))

            
                result_path = os.path.join(img_dir, 'out', str(''.join(fname)))
                img = transforms.ToPILImage()(self.denorm(fake).squeeze().cpu().data)
                img.save(result_path)

                result_path = os.path.join(img_dir, 'mst', str(''.join(fname)))
                img = transforms.ToPILImage()(img_concat.squeeze().cpu().data)
                img.save(result_path)


                
        return np.mean(ssim_list), np.mean(sip_list), np.mean(psnr_list)

    def test(self):
        self.restore_model()
        f1 = open(os.path.join(self.config.output_dir, 'test_ssim.txt'), 'a+')
        f2 = open(os.path.join(self.config.output_dir, 'test_sip.txt'), 'a+')
        f3 = open(os.path.join(self.config.output_dir, 'test_psnr.txt'), 'a+')
        
      
        name = self.config.main_dir
        print('{}\n'.format(str(name)))
        data_loader = get_loader_test(self.config, name)
        mean_ssim, mean_sip, mean_psnr = self.compute_mean(data_loader, name) 
        f1.write(str(mean_ssim.item())+'\t')
        f2.write(str(mean_sip.item())+'\t')
        f3.write(str(mean_psnr.item())+'\t')
        f1.write('\n')
        f2.write('\n')
        f3.write('\n')
       

def get_loader_test(config, data_dir):
    transform = []
    transform.append(T.Resize((128, 128)))
    transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform1 = T.Compose(transform)

    dataset = Data_Read(config.main_dir, transform1, data_dir)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=0,
                                  num_workers=1)
    return data_loader


class Data_Read(data.Dataset):
    def __init__(self, path_main, transform1, mode):

        self.path_mixture = path_main
        self.transform1 = transform1

        files = os.listdir(self.path_main)
        self.num_images = len(files)

        self.datapair = []
        for file in files:
        
            self.datapair.append([file])

    def __getitem__(self, index):
        filename = self.datapair[index]
        img_mix = Image.open(os.path.join(self.path_mixture, filename)).convert('RGB')
        
        return self.transform1(img_mix), filename

    def __len__(self):
        return self.num_images


def main(config):
    if config.test_iter != 0:
        config.output_dir = config.output_dir + str(config.test_iter)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    cudnn.benchmark = True
    solver = Solver(config)
    solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, default='/home/qzheng/nips/dataset/test/ly220/syn/')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--save_img', type=int, default=0)
    parser.add_argument('--test_iter', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='test')
    config = parser.parse_args()
    print(config)
    main(config)





