'''
build model for CIFAR-10
Copyright (c) Xiangzi Dai, 2019
'''
import torch
from nn import MLPConcatLayer,ConvConcatLayer
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as Weight_norm
import numpy as np
import plotting
class _G(nn.Module):
	def __init__(self,num_classes): #inputs (batch,c,w,h)
                super(_G,self).__init__()
                self.num = num_classes
                self.mlp = MLPConcatLayer(self.num)
                self.linear = nn.Linear(100+self.num,4*4*512)
                self.concat = ConvConcatLayer(self.num)
                self.main = nn.Sequential(
                nn.BatchNorm2d(512),
                nn.ReLU(True))
                
                self.main2 = nn.Sequential( #batch,512,8,8
                nn.ConvTranspose2d(522,256,4,2,1,bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True))

                self.main3 = nn.Sequential( #batch,512,16,16
                nn.ConvTranspose2d(266,128,4,2,1,bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True))

                self.main4 = nn.Sequential( #batch,3,32,32
                nn.ConvTranspose2d(138,3,4,2,1,bias=False),
		nn.Tanh()
		)
	def forward(self,inputs,gen_y):
                inputs = self.mlp(inputs,gen_y) #batch,4*4*512
                inputs  = self.linear(inputs)
            
                inputs = inputs.reshape(inputs.shape[0],512,4,4) #batch,512,4,4
                inputs = self.main(inputs) # batch,512,4,4
                inputs = self.concat(inputs,gen_y)#batch,512+self.num,4,4
                inputs = self.main2(inputs)#batch,256,8,8
                inputs = self.concat(inputs,gen_y)#batch,256+self.num,8,8
                inputs = self.main3(inputs)#batch,128,16,16
                inputs = self.concat(inputs,gen_y)#batch,128+self.num,16,16
                output = self.main4(inputs)#batch,3,32,32
                return output
class _D(nn.Module):
        def __init__(self,num_classes):
                super(_D,self).__init__()
                self.main = nn.Sequential(
                        nn.Conv2d(3, 96, 3, 1, 1, bias=False), #32*32
                        #nn.BatchNorm2d(96),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(96, 96, 3, 1, 1, bias=False), #32*32
                        #nn.BatchNorm2d(96),
                        nn.LeakyReLU(0.2),

                        nn.Conv2d(96, 96, 3, 2, 1, bias=False), #16*16
                        #nn.BatchNorm2d(96),
                        nn.LeakyReLU(0.2),
                        nn.Dropout2d(p=0.5),

                        nn.Conv2d(96, 192, 3, 1, 1, bias=False), #16*16
                        #nn.BatchNorm2d(192),
                        nn.LeakyReLU(0.2),

                        nn.Conv2d(192, 192, 3, 1, 1, bias=False), #16*16
                        #nn.BatchNorm2d(192),
                        nn.LeakyReLU(0.2),

                        nn.Conv2d(192, 192, 3, 2, 1, bias=False), #8*8
                        #nn.BatchNorm2d(192),
                        nn.LeakyReLU(0.2),
                        nn.Dropout2d(p=0.5),

                        nn.Conv2d(192, 192, 3, 1, 0, bias=False), #6*6
                        #nn.BatchNorm2d(192),
                        nn.LeakyReLU(0.2),

                        nn.Conv2d(192, 192, 1, 1, 0, bias=False), #6*6
                        #nn.BatchNorm2d(192),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(192, 192, 1, 1, 0, bias=False), #6*6
                        #nn.BatchNorm2d(192),
                        nn.LeakyReLU(0.2),
                        )
                self.main2 = nn.Sequential(
                    nn.Linear(192,2*num_classes)
                        )
        def forward(self,inputs,feature=False):
                inputs = inputs+torch.normal(mean=torch.zeros((inputs.shape)),std=0.2).cuda()

                output = self.main(inputs)# 192*6*6
                output = F.adaptive_avg_pool2d(output,[1,1])#192*1*1
                if feature:
                    return output
                output = output.view(-1,192)#1*192
                features = self.main2(output)#2*10
                return torch.squeeze(features) #20
class Train(object):
    def __init__(self,G,D,G_optim,D_optim,num_classes):
        self.G = G
        self.D = D
        self.G_optim = G_optim
        self.D_optim = D_optim
        self.loss = nn.CrossEntropyLoss()
        self.noise = torch.randn(100, 100)
        self.num = num_classes
    def log_sum_exp(self,x, axis = 1):
        m = torch.max(x, dim = 1)[0]
        return torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))+m
    def train_batch_disc(self,x,y,gen_y,weight_gen_loss):
        noise = torch.randn(y.shape[0], 100)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x = x.cuda()
            gen_y = gen_y.cuda()
            y = y.cuda()

        self.D.train()
        self.G.train()

        lab = self.D(x)
        gen = self.G(noise,gen_y)
        gen_data = self.D(gen)
        

        #(batch,2,self.num) ---< (batch,2)
        source_lab = torch.matmul(torch.reshape(lab,(y.shape[0],2,self.num)),F.one_hot(y,self.num).unsqueeze(-1).float().cuda()).squeeze()
        source_gen = torch.matmul(torch.reshape(gen_data,(gen_y.shape[0],2,self.num)),F.one_hot(gen_y,self.num).unsqueeze(-1).float().cuda()).squeeze()
        #real and fake class loss
        #loss_source_gen = self.loss(source_gen,torch.zeros(gen_y.shape[0]))
        loss_source_lab = self.loss(source_lab,torch.zeros(y.shape[0]).long().cuda())+self.loss(source_gen,torch.ones(y.shape[0]).long().cuda())

        #(batch,2,sel.num) ---<(batch,self.num)
        class_lab = torch.matmul(torch.reshape(lab,(y.shape[0],2,self.num)).permute(0,2,1),torch.ones(y.shape[0],2,1).float().cuda()).squeeze() #(batch,self.num)
        class_gen = torch.matmul(torch.reshape(gen_data,(gen_y.shape[0],2,self.num)).permute(0,2,1),torch.ones(gen_y.shape[0],2,1).float().cuda()).squeeze()
        # real class loss
        loss_class_lab = self.loss(class_lab,y)
        loss_class_gen = self.loss(class_gen,gen_y)

        
        #total loss
        loss_lab = (1-weight_gen_loss)*loss_source_lab+weight_gen_loss*(loss_class_lab+loss_class_gen)
        #acc
        acc = torch.mean((class_lab.max(1)[1] == y).float())
        self.D_optim.zero_grad()
        loss_lab.backward()
        self.D_optim.step()
        return loss_lab.item(),acc.item()

    def train_batch_gen(self,x,gen_y,weight_gen_loss):
        noise = torch.randn(gen_y.shape[0], 100)
        if torch.cuda.is_available():
            noise = noise.cuda()
            x = x.cuda()

        self.D.train()
        self.G.train()

        gen = self.G(noise,gen_y)
        gen_data = self.D(gen)

        output_x = self.D(x,feature=True)
        output_gen = self.D(gen,feature=True)
        #gen loss
        source_gen = torch.matmul(torch.reshape(gen_data,(gen_y.shape[0],2,self.num)),F.one_hot(gen_y,self.num).float().cuda().unsqueeze(-1)).squeeze()
        loss_source_gen = self.loss(source_gen,torch.zeros(gen_y.shape[0]).long().cuda())
        #feature loss
        m1 = torch.mean(output_x,dim=0)
        m2 = torch.mean(output_gen,dim=0)
        feature_loss = torch.mean(torch.abs(m1-m2))
        #total loss
        total_loss = (1-weight_gen_loss)*(loss_source_gen+0.5*feature_loss)
        self.G_optim.zero_grad()
        self.D_optim.zero_grad()
        total_loss.backward()
        self.G_optim.step()
        return total_loss.item()
        
    def test(self,x,y):
        self.D.eval()
        self.G.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                x,y = x.cuda(),y.cuda()
            output = self.D(x)
            output = torch.matmul(torch.reshape(output,(y.shape[0],2,self.num)).permute(0,2,1),torch.ones(y.shape[0],2,1).cuda()).squeeze()
            acc = torch.mean((output.max(1)[1] == y).float())
        return acc.item()
    def save_png(self,save_dir,epoch):
        y = torch.from_numpy(np.int32(np.repeat(np.arange(10).reshape(10,),10,axis=0)))
        if torch.cuda.is_available():
            noise = self.noise.cuda()
            y = y.long().cuda()
        with torch.no_grad():
            gen_data = self.G(noise,y)
        gen_data = gen_data.cpu().detach().numpy()
        img_bhwc = np.transpose(gen_data, (0, 2, 3, 1))
        img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title='CIFAR10 samples '+str(epoch)+" epochs")
        plotting.plt.savefig(save_dir+"cifar_sample_feature_match_"+str(epoch)+".png")

