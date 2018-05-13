# coding:utf-8
'''
python 3.5
pytorch 0.4.0
visdom 0.1.7
torchnet 0.0.2
auther: helloholmes
'''
import os
import torch
import torchvision as tv
import itchat
import time
from torch import nn
from tqdm import tqdm
from torchnet import meter
from models import model
from torchvision import transforms as T
from torch.utils.data import DataLoader
from utils.visualize import Visualizer
from config import DefaultConfig
from torch.autograd import Variable

def itchat_send(msg):
    itchat.send(msg=msg+'\n'+time.asctime(time.localtime(time.time())), toUserName='filehelper')

def dataloader(opt):
    transform = T.Compose([
                    T.Resize(opt.image_size),
                    T.CenterCrop(opt.image_size),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))])
    trainset = tv.datasets.ImageFolder(opt.root, transform=transform)
    dataloader = DataLoader(trainset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            drop_last=True)
    return dataloader

def train(opt):
    model_G = getattr(model, opt.G_model)(opt)
    model_D = getattr(model, opt.D_model)(opt)

    vis = Visualizer(opt.env)

    if opt.load_model_path:
        pass

    train_dataloder = dataloader(opt)

    criterion = torch.nn.BCELoss()
    lr_g = opt.lr_g
    lr_d = opt.lr_d
    optimizer_g = torch.optim.Adam(model_G.parameters(), lr_g, betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(model_D.parameters(), lr_d, betas=(opt.beta1, 0.999))

    # label
    true_labels = torch.ones(opt.batch_size)
    fake_labels = torch.zeros(opt.batch_size)
    fix_noises = Variable(torch.randn((opt.batch_size, opt.nz, 1, 1)))
    noises = Variable(torch.randn((opt.batch_size, opt.nz, 1, 1)))

    # meter
    loss_G_meter = meter.AverageValueMeter()
    loss_D_meter = meter.AverageValueMeter()

    if opt.use_gpu:
        model_G.cuda()
        model_D.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()

    for epoch in range(opt.max_epoch):
        loss_G_meter.reset()
        loss_D_meter.reset()

        for ii, (real_img, _) in tqdm(enumerate(train_dataloder)):
            if opt.use_gpu:
                real_img = real_img.cuda()

            if ii % opt.d_every == 0:
                # train distinguish network
                optimizer_d.zero_grad()
                # train by real image
                output = model_D(real_img)
                loss_d_real = criterion(output, true_labels)
                loss_d_real.backward()

                # train by fake image
                # refresh the value of noises
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = model_G(noises).detach()
                output = model_D(fake_img)
                loss_d_fake = criterion(output, fake_labels)
                loss_d_fake.backward()

                optimizer_d.step()
                loss_d = loss_d_fake + loss_d_real
                loss_D_meter.add(loss_d.item())

            if ii % opt.g_every == 0:
                # train generate network
                optimizer_g.zero_grad()
                # train by fake image
                # refresh the value of noises
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = model_G(noises)
                output = model_D(fake_img)

                loss_g = criterion(output, true_labels)
                loss_g.backward()

                optimizer_g.step()
                loss_G_meter.add(loss_g.item())

            if ii % opt.print_freq:
                vis.plot('loss_d', loss_D_meter.value()[0])
                vis.plot('loss_g', loss_G_meter.value()[0])
                fix_fake_img = model_G(fix_noises)
                vis.images(fix_fake_img.data.cpu().numpy()[:64]*0.5+0.5, win='fixfake')

        if (epoch+1)%20 == 0:
            model_G.save(opt.save_model_path+opt.G_model+'_'+str(epoch))
            model_D.save(opt.save_model_path+opt.D_model+'_'+str(epoch))

if __name__ == '__main__':
    opt = DefaultConfig()
    opt.parse({'max_epoch': 100})

    train(opt)
