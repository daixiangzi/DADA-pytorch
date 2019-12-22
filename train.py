'''
Training script for CIFAR-10
Copyright (c) Xiangzi Dai, 2019
'''
from __future__ import print_function

import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cifar10_data
from tensorboardX import SummaryWriter
from config import Config
from Nets import _G,_D,Train
import numpy as np
from PIL import Image
import torchvision.utils as vutils
def gen_minibatches(x,y,batch_size,shuffle=False):
    assert len(x) == len(y), "Training data size don't match"
    if shuffle:
        ids = np.random.permutation(len(x))
    else:
        ids = np.arange(len(x))
    for start_idx in range(0,len(x)-batch_size+1,batch_size):
        ii = ids[start_idx:start_idx+batch_size]
        yield x[ii],y[ii]

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2') != -1 or classname.find('ConvTranspose2d')!= -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)

opt = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
use_cuda = torch.cuda.is_available()
writer = SummaryWriter(log_dir=opt.logs)
# Random seed
if opt.seed is None:
    opt.seed = random.randint(1, 10000)
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if use_cuda:
    torch.cuda.manual_seed_all(opt.seed)


def main():
    if not os.path.isdir(opt.save_img):
        os.mkdir(opt.save_img)
    if not os.path.isdir(opt.logs):
        os.mkdir(opt.logs)
    if not os.path.isdir(opt.data_dir):
        os.mkdir(opt.data_dir)

    # Data
    trainx, trainy = cifar10_data.load(opt.data_dir, subset='train')
    testx, testy = cifar10_data.load(opt.data_dir, subset='test')
    
    # Model
    G = _G(num_classes=opt.num_classes)
    D = _D(num_classes=opt.num_classes)
    if use_cuda:
        D = torch.nn.DataParallel(D).cuda()
        G = torch.nn.DataParallel(G).cuda()
        cudnn.benchmark = True
    D.apply(weights_init)
    G.apply(weights_init)
    print('    G params: %.2fM,D params: %.2fM' % (sum(p.numel() for p in G.parameters())/1000000.0,sum(p.numel() for p in D.parameters())/1000000.0))
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    T = Train(G,D,optimizerG,optimizerD,opt.num_classes)
    #data shffule
    train = {}
    for i in range(10):
        train[i] = trainx[trainy == i][:opt.count]
    y_data = np.concatenate([trainy[trainy == i][:opt.count] for i in range(10)], axis=0)
    x_data = np.concatenate([train[i] for i in range(10)], axis=0)
    ids = np.arange(x_data.shape[0])
    np.random.shuffle(ids)
    trainx = x_data[ids]
    trainy = y_data[ids]
    nr_batches_train = int(trainx.shape[0] / opt.train_batch_size)
    nr_batches_test = int(testx.shape[0] / opt.test_batch_size)
    # Train
    best_acc = 0.0
    weight_gen_loss = 0.0
    for epoch in range(opt.epochs):
        D_loss,Train_acc = 0.0,0.0
        if epoch == opt.G_epochs:
            weight_gen_loss = 1.0
        if epoch < opt.G_epochs:
            for x_batch, y_batch in gen_minibatches(trainx, trainy, batch_size=opt.train_batch_size,shuffle=True):
                gen_y = torch.from_numpy(np.int32(np.random.choice(opt.num_classes, (y_batch.shape[0],)))).long()
                x_batch = torch.from_numpy(x_batch)
                y_batch = torch.from_numpy(y_batch).long()
                d_loss,train_acc = T.train_batch_disc(x_batch, y_batch, gen_y,weight_gen_loss)
                D_loss += d_loss
                Train_acc += train_acc
                for j in range(2):
                    gen_y_ = y_batch
                    genloss = T.train_batch_gen(x_batch,gen_y_,weight_gen_loss)
        else:
            for x_batch, y_batch in gen_minibatches(trainx, trainy, batch_size=opt.train_batch_size,shuffle=True):
                gen_y = torch.from_numpy(np.int32(np.random.choice(opt.num_classes, (y_batch.shape[0],)))).long()
                x_batch = torch.from_numpy(x_batch)
                y_batch = torch.from_numpy(y_batch).long()
                d_loss,train_acc = T.train_batch_disc(x_batch, y_batch, gen_y,weight_gen_loss)
                D_loss += d_loss
                Train_acc += train_acc
        D_loss /= nr_batches_train
        Train_acc /= nr_batches_train
    # test
        test_acc = 0.0
        for x_batch, y_batch in gen_minibatches(testx, testy, batch_size=opt.test_batch_size,shuffle=False):
            x_batch = torch.from_numpy(x_batch)
            y_batch = torch.from_numpy(y_batch).long()
            test_acc += T.test(x_batch, y_batch)
        test_acc /= nr_batches_test
        if test_acc >best_acc:
            best_acc = test_acc
            #save gen img
        T.save_png(opt.save_img,epoch)
        if (epoch+1)%(opt.fre_print)==0:
            print("Iteration %d, D_loss = %.4f,train acc = %.4f, test acc = %.4f,best acc = %.4f" % (epoch,D_loss,Train_acc, test_acc,best_acc))        
        #viso
        writer.add_scalar('train/D_loss',D_loss,epoch)
        #writer.add_scalar('train/un_loss_supervised',total_unlab,epoch)
        #writer.add_scalar('train/gen_loss',total_gen,epoch)
        writer.add_scalar('train/acc',Train_acc,epoch)
        writer.add_scalar('test/acc',test_acc,epoch)


if __name__ == '__main__':
    main()
