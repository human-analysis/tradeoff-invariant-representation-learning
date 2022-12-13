# net.py

import torch.nn as nn
import torch.nn.functional as F
import torch
__all__ = ['EncDecGaussian', 'EncDecAdvGaussian', 'Gaussian', 'EncGaussian' ,'AdvGaussian']


class EncGaussian(nn.Module):

    def __init__(self, ndim, r, hdl_enc):
        super(EncGaussian, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, hdl_enc),
            nn.PReLU(),
            nn.Linear(hdl_enc, hdl_enc),
            nn.PReLU(),
            nn.Linear(hdl_enc, hdl_enc),
            nn.PReLU(),
            nn.Linear(hdl_enc, r)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-16)
        return z

class EncDecGaussian(nn.Module):
    
    def __init__(self, ndim, nout, r, hdl_enc, hdl_tgt):
        super(EncDecGaussian, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, hdl_enc),
            nn.PReLU(),
            nn.Linear(hdl_enc, hdl_enc),
            nn.PReLU(),
            nn.Linear(hdl_enc, hdl_enc),
            nn.PReLU(),
            nn.Linear(hdl_enc, r)
        )

        self.decoder = nn.Sequential(
            nn.Linear(r, hdl_tgt),
            nn.PReLU(),
            nn.Linear(hdl_tgt, hdl_tgt),
            # nn.PReLU(),
            # nn.Linear(hdl_tgt, hdl_tgt),
            nn.PReLU(),
            nn.Linear(hdl_tgt, nout),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-16)
        out = self.decoder(z)
        return z, out

class AdvGaussian(nn.Module):

    def __init__(self, nout, r, hdl):
        super(AdvGaussian, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out


class Gaussian(nn.Module):
    def __init__(self, r, hdl, nout):
        super(Gaussian, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl),
            # nn.PReLU(),
            # nn.Linear(hdl_tgt, hdl_tgt),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )


    def forward(self, x):
        out = self.decoder(x)
        return out

class EncDecAdvGaussian(nn.Module):

    def __init__(self, ndim, r, hdl):
        super(EncDecAdvGaussian, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, hdl),
            nn.PReLU(),
            nn.BatchNorm1d(hdl),
            nn.Linear(hdl, int(hdl/2)),
            nn.PReLU(),
            nn.BatchNorm1d(int(hdl/2)),
            nn.Linear(int(hdl/2), r, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.Linear(r, hdl, bias=True),
            nn.PReLU(),
            nn.BatchNorm1d(hdl),
            nn.Dropout(p=0.5),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.BatchNorm1d(hdl),
            nn.Linear(hdl, ndim)
        )
        self.adversary = nn.Sequential(
            nn.Linear(r, hdl, bias=True),
            nn.PReLU(),
            nn.BatchNorm1d(hdl),
            nn.Dropout(p=0.5),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.BatchNorm1d(hdl),
            nn.Linear(hdl, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        out_adv = self.adversary(z)
        return z, out, out_adv
