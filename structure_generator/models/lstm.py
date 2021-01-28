import torch
import torch.nn as nn
import models.convolutional_rnn as crnn 

ConvLSTMCell = crnn.Conv2dLSTMCell


class DeterministicConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_layers=2, batch_size=16, multiplier=1):
        super(DeterministicConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.nf = nf = hidden_channels*multiplier

        self.embed = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.lstm = nn.ModuleList([ConvLSTMCell(nf, nf, 3) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Conv2d(nf, out_channels, 3, 1, 1),
                nn.ReLU(True),
                )

    def init_hidden(self, bs, h, w, device):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(bs, self.nf, h, w, device=device),
                           torch.zeros(bs, self.nf, h, w, device=device)))
        return hidden

    def forward(self, input, hidden=None):
        embedded = self.embed(input)
        h_in = embedded

        if hidden is None:
            hidden = self.init_hidden(h_in.size(0), h_in.size(2), h_in.size(3), h_in.device)

        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        out = self.output(h_in)

        return out, hidden


class GaussianConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_layers=1, batch_size=16, multiplier=1):
        super(GaussianConvLSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.nf = nf = hidden_channels*multiplier

        self.embed = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.lstm = nn.ModuleList([ConvLSTMCell(nf, nf, 3) for i in range(self.n_layers)])
        self.mu_net = nn.Sequential(nn.Conv2d(nf, out_channels, 3, 1, 1))
        self.logvar_net = nn.Sequential(nn.Conv2d(nf, out_channels, 3, 1, 1))

    def init_hidden(self, bs, h, w, device):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(bs, self.nf, h, w, device=device),
                           torch.zeros(bs, self.nf, h, w, device=device)))
        return hidden

    def reparameterize(self, mu, logvar):
        sigma = logvar.mul(0.5).exp_()
        eps = torch.randn_like(sigma)
        return eps.mul(sigma).add_(mu)

    def forward(self, input, hidden=None):
        embedded = self.embed(input)
        h_in = embedded

        if hidden is None:
            hidden = self.init_hidden(h_in.size(0), h_in.size(2), h_in.size(3), h_in.device)

        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, hidden
