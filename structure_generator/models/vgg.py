import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_onehot(x, n_class):
    """
    Args:
        x: torch.LongTensor (B, 1, H, W)
    Return:
        torch.FloatTensor(B, C, H, W)
    """
    # one-liner but a little bit slow
    # onehot = F.one_hot(x.squeeze(1), self.input_dim).permute(0, 3, 1, 2)

    # faster version
    B, _, H, W = x.size()
    C = n_class
    onehot = torch.zeros([B, C, H*W], device=x.device, requires_grad=False)
    onehot.scatter_(1, x.long().view([B, 1, -1]), 1)
    onehot = onehot.reshape([B, C, H, W])
    return onehot


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', use_bn=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        if use_bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = Identity()

        if act == 'relu':
            self.act = nn.ReLU(True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif act == 'logsoftmax':
            self.act = nn.LogSoftmax()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'none':
            self.act = Identity()
        else:
            raise ValueError()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Encoder64(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):
        super(Encoder64, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multiplier = multiplier
        self.nf = nf = hidden_dim*multiplier

        self.c1 = nn.Sequential(
                ConvLayer(input_dim, nf*1, 'relu', use_bn),
                ConvLayer(nf*1, nf*1, 'relu', use_bn),
                )
        self.c2 = nn.Sequential(
                ConvLayer(nf*1, nf*2, 'relu', use_bn),
                ConvLayer(nf*2, nf*2, 'relu', use_bn),
                )
        self.c3 = nn.Sequential(
                ConvLayer(nf*2, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                )
        self.c4 = nn.Sequential(
                ConvLayer(nf*4, output_dim, output_act, False)
                )
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        if x.size() == torch.Size([x.size(0), 1, x.size(2), x.size(3)]):
            x = make_onehot(x, self.input_dim)
        elif x.size() == torch.Size([x.size(0), self.input_dim, x.size(2), x.size(3)]):
            x = x
        else:
            raise
        h1 = self.c1(x)
        h2 = self.c2(self.down(h1))
        h3 = self.c3(self.down(h2))
        h4 = self.c4(self.down(h3))
        return h4, [h1, h2, h3]


class Decoder64(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):
        super(Decoder64, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multiplier = multiplier
        self.nf = nf = hidden_dim*multiplier

        self.c1 = nn.Sequential(
                ConvLayer(input_dim + nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*2, 'relu', use_bn),
                )
        self.c2 = nn.Sequential(
                ConvLayer(nf*2*2, nf*2, 'relu', use_bn),
                ConvLayer(nf*2, nf*1, 'relu', use_bn),
                )
        self.c3 = nn.Sequential(
                ConvLayer(nf*1*2, nf*1, 'relu', use_bn),
                ConvLayer(nf*1, output_dim, output_act, False),
                )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, data):
        x, skip = data
        h1 = self.c1(torch.cat([self.up(x), skip[-1]], dim=1))
        h2 = self.c2(torch.cat([self.up(h1), skip[-2]], dim=1))
        h3 = self.c3(torch.cat([self.up(h2), skip[-3]], dim=1))
        return h3


class Encoder128(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):
        super(Encoder128, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multiplier = multiplier
        self.nf = nf = hidden_dim*multiplier

        self.c1 = nn.Sequential(
                ConvLayer(input_dim, nf*1, 'relu', use_bn),
                ConvLayer(nf*1, nf*1, 'relu', use_bn),
                )
        self.c2 = nn.Sequential(
                ConvLayer(nf*1, nf*2, 'relu', use_bn),
                ConvLayer(nf*2, nf*2, 'relu', use_bn),
                )
        self.c3 = nn.Sequential(
                ConvLayer(nf*2, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                )
        self.c4 = nn.Sequential(
                ConvLayer(nf*4, nf*8, 'relu', use_bn),
                ConvLayer(nf*8, nf*8, 'relu', use_bn),
                ConvLayer(nf*8, nf*8, 'relu', use_bn),
                )
        self.c5 = nn.Sequential(
                ConvLayer(nf*8, output_dim, output_act, False)
                )
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        if x.size() == torch.Size([x.size(0), 1, x.size(2), x.size(3)]):
            x = make_onehot(x, self.input_dim)
        elif x.size() == torch.Size([x.size(0), self.input_dim, x.size(2), x.size(3)]):
            x = x
        else:
            raise
        h1 = self.c1(x)
        h2 = self.c2(self.down(h1))
        h3 = self.c3(self.down(h2))
        h4 = self.c4(self.down(h3))
        h5 = self.c5(self.down(h4))
        return h5, [h1, h2, h3, h4]


class Decoder128(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):
        super(Decoder128, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multiplier = multiplier
        self.nf = nf = hidden_dim*multiplier

        self.c1 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(input_dim + nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                   
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*8, nf*4, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c2 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*4*2, nf*4, 'relu', use_bn),                                                                                                                                                                             
                ConvLayer(nf*4, nf*4, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*4, nf*2, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c3 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*2*2, nf*2, 'relu', use_bn),                                                                                                                                                                             
                ConvLayer(nf*2, nf*1, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c4 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*1*2, nf*1, 'relu', use_bn),                                                                                                                                                                             
                ConvLayer(nf*1, output_dim, output_act, False),                                                                                                                                                                      
                )  
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, data):
        x, skip = data
        h1 = self.c1(torch.cat([self.up(x), skip[-1]], dim=1))
        h2 = self.c2(torch.cat([self.up(h1), skip[-2]], dim=1))
        h3 = self.c3(torch.cat([self.up(h2), skip[-3]], dim=1))
        h4 = self.c4(torch.cat([self.up(h3), skip[-4]], dim=1))
        return h4


class Encoder256(nn.Module):                                                                                                                                                                                                    
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):                                                                                                  
        super(Encoder256, self).__init__()                                                                                                                                                                                      
        self.input_dim = input_dim                                                                                                                                                                                                   
        self.hidden_dim = hidden_dim                                                                                                                                                                                                 
        self.output_dim = output_dim                                                                                                                                                                                                 
        self.multiplier = multiplier                                                                                                                                                                                                 
        self.nf = nf = hidden_dim*multiplier                                                                                                                                                                                         
                                                                                                                                                                                                                                     
        self.c1 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(input_dim, nf*1, 'relu', use_bn),                                                                                                                                                                          
                ConvLayer(nf*1, nf*1, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c2 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*1, nf*2, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*2, nf*2, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c3 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*2, nf*4, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*4, nf*4, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*4, nf*4, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c4 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*4, nf*8, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c5 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c6 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*8, output_dim, output_act, False)                                                                                                                                                                       
                )                                                                                                                                                                                                                    
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)                                                                                                                                                                 
                                                                                                                                                                                                                                     
    def forward(self, x):                                                                                                                                                                                                            
        if x.size() == torch.Size([x.size(0), 1, x.size(2), x.size(3)]):                                                                                                                                                         
            x = make_onehot(x, self.input_dim)                                                                                                                                                                                   
        elif x.size() == torch.Size([x.size(0), self.input_dim, x.size(2), x.size(3)]):                                                                                                                                          
            x = x                                                                                                                                                                                                                
        else:                                                                                                                                                                                                                    
            raise                                                                                                                                                                                                                
        h1 = self.c1(x)                                                                                                                                                                                                              
        h2 = self.c2(self.down(h1))                                                                                                                                                                                                  
        h3 = self.c3(self.down(h2))                                                                                                                                                                                                  
        h4 = self.c4(self.down(h3))                                                                                                                                                                                                  
        h5 = self.c5(self.down(h4))                                                                                                                                                                                                  
        h6 = self.c6(self.down(h5))                                                                                                                                                                                                  
        return h6, [h1, h2, h3, h4, h5]


class Decoder256(nn.Module):                                                                                                                                                                                                    
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):                                                                                                                      
        super(Decoder256, self).__init__()                                                                                                                                                                                      
        self.input_dim = input_dim                                                                                                                                                                                                   
        self.hidden_dim = hidden_dim                                                                                                                                                                                                 
        self.output_dim = output_dim                                                                                                                                                                                                 
        self.multiplier = multiplier                                                                                                                                                                                                 
        self.nf = nf = hidden_dim*multiplier                                                                                                                                                                                         
                                                                                                                                                                                                                                     
        self.c1 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(input_dim + nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                   
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c2 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*8*2, nf*8, 'relu', use_bn),                                                                                                                                                                             
                ConvLayer(nf*8, nf*8, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*8, nf*4, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c3 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*4*2, nf*4, 'relu', use_bn),                                                                                                                                                                             
                ConvLayer(nf*4, nf*4, 'relu', use_bn),                                                                                                                                                                               
                ConvLayer(nf*4, nf*2, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c4 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*2*2, nf*2, 'relu', use_bn),                                                                                                                                                                             
                ConvLayer(nf*2, nf*1, 'relu', use_bn),                                                                                                                                                                               
                )                                                                                                                                                                                                                    
        self.c5 = nn.Sequential(                                                                                                                                                                                                     
                ConvLayer(nf*1*2, nf*1, 'relu', use_bn),                                                                                                                                                                             
                ConvLayer(nf*1, output_dim, output_act, False),                                                                                                                                                                      
                )                                                                                                                                                                                                                    
        self.up = nn.Upsample(scale_factor=2, mode='nearest')                                                                                                                                                                        
                                                                                                                                                                                                                                     
    def forward(self, data):                                                                                                                                                                                                         
        x, skip = data                                                                                                                                                                                                               
        h1 = self.c1(torch.cat([self.up(x), skip[-1]], dim=1))                                                                                                                                                                       
        h2 = self.c2(torch.cat([self.up(h1), skip[-2]], dim=1))                                                                                                                                                                      
        h3 = self.c3(torch.cat([self.up(h2), skip[-3]], dim=1))                                                                                                                                                                      
        h4 = self.c4(torch.cat([self.up(h3), skip[-4]], dim=1))                                                                                                                                                                      
        h5 = self.c5(torch.cat([self.up(h4), skip[-5]], dim=1))                                                                                                                                                                      
        return h5 


class Encoder128x256(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):
        super(Encoder128x256, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multiplier = multiplier
        self.nf = nf = hidden_dim*multiplier

        self.c1 = nn.Sequential(
                ConvLayer(input_dim, nf*1, 'relu', use_bn),
                ConvLayer(nf*1, nf*1, 'relu', use_bn),
                )
        self.c2 = nn.Sequential(
                ConvLayer(nf*1, nf*2, 'relu', use_bn),
                ConvLayer(nf*2, nf*2, 'relu', use_bn),
                )
        self.c3 = nn.Sequential(
                ConvLayer(nf*2, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                )
        self.c4 = nn.Sequential(
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                )
        self.c5 = nn.Sequential(
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                )
        self.c6 = nn.Sequential(
                ConvLayer(nf*4, output_dim, output_act, False)
                )
        self.down = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


    def forward(self, x):
        if x.size() == torch.Size([x.size(0), 1, x.size(2), x.size(3)]):
            x = make_onehot(x, self.input_dim)
        elif x.size() == torch.Size([x.size(0), self.input_dim, x.size(2), x.size(3)]):
            x = x
        else:
            raise
        h1 = self.c1(x)
        h2 = self.c2(self.down(h1))
        h3 = self.c3(self.down(h2))
        h4 = self.c4(self.down(h3))
        h5 = self.c5(self.down(h4))
        h6 = self.c6(h5)
        return h6, [h1, h2, h3, h4, h5]


class Decoder128x256(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=128, multiplier=1, output_act='none', use_bn=True):
        super(Decoder128x256, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.multiplier = multiplier
        self.nf = nf = hidden_dim*multiplier


        self.c1 = nn.Sequential(
                ConvLayer(input_dim + nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                )
        self.c2 = nn.Sequential(
                ConvLayer(nf*4*2, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                )
        self.c3 = nn.Sequential(
                ConvLayer(nf*4*2, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*4, 'relu', use_bn),
                ConvLayer(nf*4, nf*2, 'relu', use_bn),
                )
        self.c4 = nn.Sequential(
                ConvLayer(nf*2*2, nf*2, 'relu', use_bn),
                ConvLayer(nf*2, nf*1, 'relu', use_bn),
                )
        self.c5 = nn.Sequential(
                ConvLayer(nf*1*2, nf*1, 'relu', use_bn),
                ConvLayer(nf*1, output_dim, output_act, False),
                )
        self.up = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, data):
        x, skip = data
        h1 = self.c1(torch.cat([x, skip[-1]], dim=1))
        h2 = self.c2(torch.cat([self.up(h1), skip[-2]], dim=1))
        h3 = self.c3(torch.cat([self.up(h2), skip[-3]], dim=1))
        h4 = self.c4(torch.cat([self.up(h3), skip[-4]], dim=1))
        h5 = self.c5(torch.cat([self.up(h4), skip[-5]], dim=1))
        return h5


def get_autoencoder(opt):
    if opt.dataset == 'KITTI_64':
        encoder = Encoder64
        decoder = Decoder64
    elif opt.dataset == 'KITTI_128':
        encoder = Encoder128
        decoder = Decoder128
    elif opt.dataset == 'KITTI_256':
        encoder = Encoder256
        decoder = Decoder256
    elif opt.dataset == 'Cityscapes_128x256':
        encoder = Encoder128x256
        decoder = Decoder128x256
    elif opt.dataset == 'Pose_64':
        encoder = Encoder64
        decoder = Decoder64
    elif opt.dataset == 'Pose_128':
        encoder = Encoder128
        decoder = Decoder128
    else:
        raise ValueError('Unknown dataset: %s' % opt.dataset)

    encoder = encoder(opt.channels, opt.ae_size, opt.g_dim, opt.K, 'relu', opt.use_bn)
    decoder = decoder(opt.g_dim, opt.ae_size, opt.channels, opt.K, 'logsoftmax', opt.use_bn)

    return encoder, decoder
