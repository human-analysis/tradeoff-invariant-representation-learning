# kernels.py

import torch

__all__ = ['GaussianKernel', 'GaussianSigma', 'GaussianKernelSigma', 'IMQ', 'LaplacianKernel']


class LaplacianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x, s):
        n_x = x.shape[0]
        n_s = s.shape[0]

        x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
        s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

        ones_x = torch.ones([1, n_x]).to(device=x.device, dtype=x.dtype)
        ones_s = torch.ones([1, n_s]).to(device=x.device, dtype=x.dtype)

        dist = (torch.mm(torch.t(x_norm), ones_s) +
             torch.mm(torch.t(ones_x), s_norm) - 2 * torch.mm(x, torch.t(s)))

        kernel = torch.exp(-torch.sqrt(torch.abs(dist)) / (self.sigma))
        return kernel


class IMQ:
    def __init__(self, sigma):
        self.c = sigma

    def __call__(self, x, s):

        n_x = x.shape[0]
        n_s = s.shape[0]

        x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
        s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

        ones_x = torch.ones([1, n_x]).to(device=x.device, dtype=x.dtype)
        ones_s = torch.ones([1, n_s]).to(device=x.device, dtype=x.dtype)
        M = torch.mm(torch.t(x_norm), ones_s) + torch.mm(torch.t(ones_x), s_norm) - 2 * torch.mm(x, torch.t(s))
        # import pdb; pdb.set_trace()
        kernel = self.c * torch.pow(M + self.c, -0.5)

        return kernel

class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x, s):

        n_x = x.shape[0]
        n_s = s.shape[0]

        x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
        s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

        ones_x = torch.ones([1, n_x]).to(device=x.device, dtype=x.dtype)
        ones_s = torch.ones([1, n_s]).to(device=s.device, dtype=s.dtype)

        kernel = torch.exp(
            (-torch.mm(torch.t(x_norm), ones_s) -
             torch.mm(torch.t(ones_x), s_norm) + 2 * torch.mm(x, torch.t(s)))
            / (2*self.sigma**2))

        return kernel

class GaussianKernelSigma:
    def __init__(self):
        pass

    def __call__(self, x, s, sigma):

        n_x = x.shape[0]
        n_s = s.shape[0]

        x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
        s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

        ones_x = torch.ones([1, n_x]).to(device=x.device, dtype=x.dtype)
        ones_s = torch.ones([1, n_s]).to(device=s.device, dtype=s.dtype)

        kernel = torch.exp(
            (-torch.mm(torch.t(x_norm), ones_s) -
             torch.mm(torch.t(ones_x), s_norm) + 2 * torch.mm(x, torch.t(s)))
            / (2*sigma**2))

        return kernel

class GaussianSigma:
    def __init__(self):
        pass

    def __call__(self, x_t, n):
        rand = torch.randperm(x_t.shape[0])
        x = x_t[rand, :]
        x = x[0: n, :]
        G = torch.sum(x*x, dim=1).unsqueeze(1)
        Q = G.repeat(1, n)
        R = G.t().repeat(n, 1)

        dists = Q + R - 2*torch.mm(x, x.t())
        dists = dists - torch.tril(dists)
        dists = torch.reshape(dists, (n**2, 1))
        sigma = torch.sqrt(0.5*torch.median(dists[dists > 0]))

        return sigma
