import pytorch_ssim
from model import GNet
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



class Solver(object):

    def __init__(self, data_loader_train, data_loader_val, data_loader_test, config):

        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dt = config.distance_type
        
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        
        self.G = GNet().to(self.device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [0.5, 0.999])
        self.D = HNet().to(self.device)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [0.5, 0.999])
        
        self.ssim_loss = pytorch_ssim.SSIM(size_average=False)
        self.ssim_si_loss = pytorch_ssim.SSIM_SI(size_average=False)
        
        self.vgg = Vgg16()
        self.init_vgg16('../models/')
        self.vgg.load_state_dict(torch.load(os.path.join('../models/', "vgg16.weight")))
        self.vgg.cuda()
        self.criterionCAE = nn.L1Loss()
        self.criterion_cross = nn.CrossEntropyLoss().cuda()
        self.BCE = nn.BCELoss()

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
        out = x #(x + 1) / 2
        return out#.clamp_(0, 1)

    def get_loss(self, batch1, batch2, dis=0):
        s1 = self.denorm(batch1)
        s2 = self.denorm(batch2)
        ssim, si = self.ssim_si_loss(s1, s2)
        perceptual_loss = self.get_perceptual_loss(batch1, batch2)
        psnr = ((s1 - s2) ** 2).mean(1).mean(1).mean(1)
        psnr = 10 * (torch.log10(1.0 / psnr))
        # if dis == 0:
        #     # reflection removal
        # loss = - ssim - si + 2 + 2 * perceptual_loss
        # score = si
        # else:
        #     # de-raining
        psnr = ((s1 - s2) ** 2).mean(1).mean(1).mean(1)
        psnr = 10 * (torch.log10(1.0 / psnr))
        loss = - ssim - si - psnr/40.0 + 3 + 2 * perceptual_loss
        # score = psnr
        return loss, ssim, perceptual_loss, psnr

    def gradient_penalty(self, y, x, t=1, t2=1):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        y = y.view(y.size(0), -1)
        #y = y.mean(dim=1)
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        t = 1    
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.mean(dydx**2, dim=1))
        t2 = 1 # 41472
        return (torch.mean((dydx_l2norm-1)**2))

    def restore_model(self, start_iters):
        G_path = os.path.join(self.config.model_save_dir, '{}-G.ckpt'.format(start_iters))
        D_path = os.path.join(self.config.model_save_dir, '{}-D.ckpt'.format(start_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        print('Loading the trained models' + G_path + D_path)

    def train(self):

        data_iter = iter(self.data_loader_train)
        fixed_mix, fixed_gro, fixed_alpha = next(data_iter)
        fixed_mix, fixed_gro, fixed_alpha = fixed_mix.to(self.device), fixed_gro.to(self.device), fixed_alpha.to(self.device).float().unsqueeze(1)
        iters_per_epoch = len(self.data_loader_train)
        start_iters = 0
        if self.config.resume_iters:
            start_iters = self.config.resume_iters
            self.restore_model(start_iters)

        g_lr = self.g_lr
        d_lr = self.d_lr
        
        
        print('Start training...')
        start_time = time.time()
        for e in range(start_iters, self.config.num_epochs):
            for i, (data_mix, data_gro, data_alpha_g) in enumerate(self.data_loader_train):
                data_mix, data_gro, data_alpha = data_mix.to(self.device), data_gro.to(self.device), data_alpha_g.to(self.device).float().unsqueeze(1)
                loss = {}
                onenumber = torch.ones([data_mix.size(0), 1], dtype=torch.float32).to(self.device)
                output_real_D = self.D(data_mix, data_alpha)
                d_loss_real, d_ssim, _, _ = self.get_loss(output_real_D, data_gro, self.dt)
                alpha_fake1, alpha_fake2, loss_sigma, _ = self.G(data_mix, data_gro, data_alpha)
                tx = (torch.rand(data_alpha.size(0), 1, 1, 1).to(self.device)-0.5)*0.2
                data_hat = (data_mix.data + tx * data_gro.data).requires_grad_(True)
                tx = torch.rand(alpha_fake1.size(0), 1).to(self.device)
                tx0 = (tx * data_alpha.data + (1 - tx) * alpha_fake1.data).requires_grad_(True)
                out_rec = self.D(data_hat, tx0)
                g_loss_gp = self.gradient_penalty(out_rec, data_hat)
                g_loss_gp = Variable(g_loss_gp.data, requires_grad=True)
                g_loss_gp = g_loss_gp

                d_loss_real = d_loss_real.mean() 
                d_loss = d_loss_real + 10 * g_loss_gp

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                loss['d_real'] = d_loss_real.item()
                loss['d_gp'] = g_loss_gp.item()
                loss['d_ssim'] = d_ssim.mean().item()
                
                # generator
                if (i + 1) % 5 == 0:
                    alpha_fake1, alpha_fake2, loss_sigma, loss5 = self.G(data_mix, data_gro, data_alpha)
                    output_fake_D = self.D(data_mix, alpha_fake1)
                    g_loss_fake, g_ssim, _, _ = self.get_loss(output_fake_D, data_gro, self.dt)
                    #print(alpha_fake.size())
                    #print(data_alpha_g.to(self.device).long().unsqueeze(1).size())

                    # print(alpha_fake1.size())
                    # print(alpha_fake2.size())
                    # print(data_alpha.size())
                    # print(onenumber.size())
                    g_entroy1 = self.BCE((alpha_fake1-0.7)/0.3, (data_alpha-0.7)/0.3)

                    g_entroy2 = self.BCE((alpha_fake2-0.7)/0.3, (onenumber-0.7)/0.3)

                    g_entroy = g_entroy1 + g_entroy2

                    g_loss_fake = g_loss_fake.mean() + 10*loss_sigma.mean()
                    g_loss = g_loss_fake + g_entroy * 10
                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    loss['g_fake'] = g_loss_fake.item()
                    loss['g_ssim'] = g_ssim.mean().item()
                    loss['g_entroy'] = g_entroy.item()
                    loss['g_sigma'] = loss_sigma.mean().item()
                    loss['g_loss5'] = loss5.mean().item()


                if (i + 1) % self.config.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                                                                             et, e + 1, self.config.num_epochs, i + 1, iters_per_epoch)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                
            with torch.no_grad():
                fixed_alpha, _, _, _ = self.G(fixed_mix, fixed_gro, fixed_alpha)
                
                fixed_fake_D = self.D(fixed_mix, fixed_alpha)
                fixed_output = self.denorm(fixed_fake_D)
                
                x_fake_list = [self.denorm(fixed_mix)]
                x_fake_list.append(self.denorm(fixed_gro))
                x_fake_list.append(fixed_output)
                
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.config.sample_dir, '{}-images.jpg'.format(e + 1))
                save_image(x_concat.data.cpu(), sample_path, nrow=1, padding=0)
                
                print('SSIM mix&gro:{}, \t SSIM:{}'.format(self.ssim_loss(self.denorm(fixed_mix), self.denorm(fixed_gro)).mean(), self.ssim_loss(fixed_output, self.denorm(fixed_gro)).mean()))
                
            
            if (e>180 or e%20 == 0):
                # Save model
                G_path = os.path.join(self.config.model_save_dir, '{}-G.ckpt'.format(e + 1))
                D_path = os.path.join(self.config.model_save_dir, '{}-D.ckpt'.format(e + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.config.model_save_dir))

            # self.val(e+1)

        if (e + 1) > (self.config.num_epoch_decay):
            g_lr -= (self.g_lr / 100000.0)
            d_lr -= (self.d_lr / 100000.0)

            for param_group in self.g_optimizer.param_groups:
                param_group['lr'] = g_lr
            for param_group in self.d_optimizer.param_groups:
                param_group['lr'] = d_lr

            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
                    
    def compute_loss1(self, data_loadert):
        loss_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        with torch.no_grad():
            for i, (mix, gro, data_alpha) in enumerate(data_loadert):
                mix, gro, data_alpha = mix.to(self.device), gro.to(self.device), data_alpha.to(self.device).float().unsqueeze(1)
                _, _, fake_out = self.G(mix, mix, data_alpha)
                fake = self.D(mix, fake_out)
                loss, score1, score2, score3 = self.get_loss(gro, fake, self.dt)
                loss_list.append(loss.mean())
                score_list1.append(score1.mean())
                score_list2.append(score2.mean())
                score_list3.append(score3.mean())
                    
        m_loss = torch.mean(torch.FloatTensor(loss_list))
        m_score1 = torch.mean(torch.FloatTensor(score_list1))
        m_score2 = torch.mean(torch.FloatTensor(score_list2))
        m_score3 = torch.mean(torch.FloatTensor(score_list3))
        return m_loss.item(), m_score1.item(), m_score2.item(), m_score3.item()
                            
    def compute_loss2(self, data_loadert):
        loss1_list = []
        loss2_list = []
        loss3_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        with torch.no_grad():
            for i, (mix, gro, data_alpha) in enumerate(data_loadert):
                mix, gro, data_alpha = mix.to(self.device), gro.to(self.device), data_alpha.to(self.device).float().unsqueeze(1)
                
                _, _, fake_out = self.G(mix, mix, data_alpha)
                fake = self.D(mix, fake_out)
                loss1, score1, _, _ = self.get_loss(gro, fake, self.dt)
                loss1_list.append(loss1.mean())
                score_list1.append(score1.mean())

                real = self.D(mix, data_alpha)
                loss2, score2, _, _ = self.get_loss(gro, real, self.dt)
                loss2_list.append(loss2.mean())
                score_list2.append(score2.mean())
    
                loss3, score3, _, _ = self.get_loss(fake, real, self.dt)
                loss3_list.append(loss3.mean())
                score_list3.append(score3.mean())
                        
                        
        m_loss1 = torch.mean(torch.FloatTensor(loss1_list))
        m_loss2 = torch.mean(torch.FloatTensor(loss2_list))
        m_loss3 = torch.mean(torch.FloatTensor(loss3_list))
        m_score1 = torch.mean(torch.FloatTensor(score_list1))
        m_score2 = torch.mean(torch.FloatTensor(score_list2))
        m_score3 = torch.mean(torch.FloatTensor(score_list3))
        
        return m_loss1.item(), m_loss2.item(), m_loss3.item(), m_score1.item(), m_score2.item(), m_score3.item()
                    
    def val(self, i):
        loss1, loss2, loss3, score1, score2, score3 = self.compute_loss2(self.data_loader_test)
        f1 = open('./ours/test.txt', 'a+')
        f1.write(str(i) + '\t' + str(loss1) + '\t' + str(loss2)+ '\t'+ str(loss3)+ '\t' + str(score1) + '\t' + str(score2)+ '\t' + str(score3) + '\n')
        print('test:\t' + str(i) + '\t' + str(loss1) + '\t' + str(loss2)+ '\t'+ str(score1) + '\t' + str(score2))
        f1.close()
            
        loss1, loss2, loss3, score1, score2, score3 = self.compute_loss2(self.data_loader_val)
        f2 = open('./ours/train.txt', 'a+')
        f2.write(str(i) + '\t' + str(loss1) + '\t' + str(loss2) + '\t' + str(loss3)+ '\t' + str(score1) + '\t' + str(score2)+ '\t' + str(score3) + '\n')
        print('train:\t' + str(i) + '\t' + str(loss1) + '\t' + str(loss2)+ '\t' + str(score1) + '\t' + str(score2))
        f2.close()
    
    def val_all(self):
        start_iters = 1
        if self.config.resume_iters:
            start_iters = self.config.resume_iters
        for e in range(start_iters, self.config.num_epochs):
            self.restore_model(e)
            self.val(e)

   