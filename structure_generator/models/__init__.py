import torch
import torch.nn as nn

from models.vgg import get_autoencoder
from models.lstm import GaussianConvLSTM, DeterministicConvLSTM


def init_weights(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d, nn.ConvTranspose1d, nn.Linear)):
        m.weight.data.normal_(0.0, 0.02)
        try:
            m.bias.data.fill_(0)
        except AttributeError:
            pass
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.normal_(1.0, 0.02)
        try:
            m.bias.data.fill_(0)
        except AttributeError:
            pass


def kl_criterion(mu1, logvar1, mu2, logvar2):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    out = kld.view(kld.size(0), -1).sum(1).mean()
    return out


__all__ = ['GaussianConvLSTM',
           'DeterministicConvLSTM',
           'get_autoencoder',
           'init_weights',
           'kl_criterion',
           ]
