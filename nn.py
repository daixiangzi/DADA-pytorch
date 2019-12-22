import torch
import torch.nn.functional as F
import torch.nn as nn
def to_hot(y,class_num):
    y = y.cpu()
    y=torch.unsqueeze(y,dim=1)
    return torch.zeros(y.shape[0],class_num).scatter_(1,y,1).cuda()

class MLPConcatLayer(nn.Module):
    def __init__(self,class_num):
        super(MLPConcatLayer,self).__init__()
        self.num = class_num
    def forward(self,x,y):
        if y.dim() == 1:
            y = F.one_hot(y,self.num)
        assert y.dim()==2,"label dim should be two"
        return torch.cat([x,y.float()],1)
        
class ConvConcatLayer(nn.Module):
    def __init__(self,class_num):
        super(ConvConcatLayer,self).__init__()
        self.num = class_num
    def forward(self,x,y):
        if y.dim() == 1:
            y = F.one_hot(y,self.num)
        if y.dim() == 2:
            y = torch.unsqueeze(torch.unsqueeze(y,-1),-1).float()
        assert y.dim()==4,"label dim should be four"
        return torch.cat([x,y*torch.ones(x.shape[0], y.shape[1], x.shape[2], x.shape[3]).cuda()],1)

