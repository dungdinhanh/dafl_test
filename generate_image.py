# Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import os
import numpy as np
import math
import sys
import pdb

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from lenet import LeNet5Half
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import resnet
import time
import torchvision.utils as vutils
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'cifar10', 'cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='cache/models/')


opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True

accr = 0
accr_best = 0


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(opt.channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


def load_model(path):
    dict_save = torch.load(path)
    generator = Generator()
    generator = nn.DataParallel(generator)
    generator.load_state_dict(dict_save['state_dict'])
    best_accr = dict_save['accuracy']
    print("Achieve best accuracy: %f"%float(best_accr))
    return generator


def save_images_diff(images, path, count):
    # method to store generated images locally
    os.makedirs(path, exist_ok=True)
    for id in range(images.shape[0]):
        image_name = str(count + id) + ".png"
        image_name = os.path.join(path, image_name)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(image_name)




def generate_images(generator: torch.nn.Module, num,path):
    generator.eval()
    count = 0
    while count < num:
        z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        gen_imgs = generator(z)
        save_images_diff(gen_imgs, path, count)
        count += opt.batch_size

def generate_batch(generator: torch.nn.Module, num_batch):
    generator.eval()
    z = Variable(torch.randn(num_batch, opt.latent_dim)).cuda()
    gen_imgs = generator(z)
    vutils.save_image(gen_imgs, "batch_dafl.png", normalize=True, scale_each=True, nrow=int(8))




if __name__ == '__main__':
    generator = load_model(os.path.join(opt.output_dir, "generator.pth"))
    generate_images(generator, 50000,"gen_images")
    generate_batch(generator, 64)
    pass


