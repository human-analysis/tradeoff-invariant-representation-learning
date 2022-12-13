import torch
import numpy as np
# from hal.metrics import Metric
# import torchmetrics.metric.Metric as Metric
import torchmetrics.metric as tm
from typing import Any, Callable, Optional
from math import pi, sqrt
import scipy.stats as scs

__all__ = ['DEPKCC']



class DEPKCC(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("zz", default=[], dist_reduce_fx=None)
        self.add_state("ss", default=[], dist_reduce_fx=None)

    def update(self, z, s, opts, rbfsigma, label=False):

        zz = z
        ss = s

        self.opts = opts
        self.rbfsigma = rbfsigma
        self.zz.append(zz)
        self.ss.append(ss)
        self.label=label

    def compute(self):

        z = torch.cat(self.zz, dim=0)
        # z_mean = torch.mean(z, dim=0)
        # z_std = torch.std(z, dim=0)
        # zz = (z - z_mean) / (z_std + 1e-10)
        zz = z
        # import pdb; pdb.set_trace()

        s = torch.cat(self.ss, dim=0)
        # s_mean = torch.mean(s, dim=0)
        # s_std = torch.std(s, dim=0)
        # ss = (s - s_mean) / (s_std + 1e-16)
        ss = s

        n = s.shape[0]


        # zz = torch.from_numpy(scs.rankdata(z.cpu().numpy(), axis=0)).float() / n
        # ss = torch.from_numpy(scs.rankdata(s.cpu().numpy(), axis=0)).float() / n
        # zz = zz.to(device=z.device)
        # ss = ss.to(device=s.device)


        #########################################################
        sigma_z_model = self.rbfsigma(z, z.shape[0])
        K_z = GaussianKernel(z, z, sigma_z_model)
        I = torch.eye(n).to(device=K_z.device)
        # K = K_z + 0.005 * zz.shape[1]/3 * I
        # K = K_z + 0.005 * I
        K = K_z + self.opts.kernel_reg * I
        model_z = torch.mm(torch.linalg.inv(K), s)
        # import pdb; pdb.set_trace()
        #######################################################

        sigma_z = self.rbfsigma(zz, zz.shape[0])
        # sigma_z1 = self.rbfsigma(zz1, zz.shape[0])
        sigma_s = self.rbfsigma(ss, ss.shape[0])
        # sigma_s1 = self.rbfsigma(ss1, ss.shape[0])

        H = torch.eye(n, dtype=ss.dtype) - torch.ones(n, dtype=ss.dtype) / n
        H = H.to(device=ss.device)

        # H_z = torch.mm(zz, zz.t())
        # K_z = GaussianKernel(zz.float(), zz.float(), sigma_z)
        # K_zm = torch.mm(H, torch.mm(K_z, H))

        K_s = GaussianKernel(ss, ss, sigma_s)
        K_sm = torch.mm(H, torch.mm(K_s, H))
        K_z = GaussianKernel(zz, zz, sigma_z)
        K_zm = torch.mm(H, torch.mm(K_z, H))

        ########################### KCC #########################################################
        D = 500
        lam = 1e-3
        Sigma_z = 1 / (sigma_z ** 2) * torch.diag(torch.ones(z.shape[1], dtype=zz.dtype).to(device=zz.device))
        Sigma_s = 1 / (sigma_s ** 2) * torch.diag(torch.ones(s.shape[1], dtype=ss.dtype).to(device=ss.device))
        Ones_z = torch.zeros(1, zz.shape[1], dtype=zz.dtype).to(device=zz.device)
        Ones_s = torch.zeros(1, ss.shape[1], dtype=ss.dtype).to(device=zz.device)

        torch.manual_seed(2)
        px = torch.distributions.MultivariateNormal(Ones_z, Sigma_z)
        torch.manual_seed(3)
        ps = torch.distributions.MultivariateNormal(Ones_s, Sigma_s)
        torch.manual_seed(4)
        p1 = torch.distributions.uniform.Uniform(torch.tensor([0.0]), 2 * torch.tensor([np.pi]))
        torch.manual_seed(5)
        p2 = torch.distributions.uniform.Uniform(torch.tensor([0.0]), 2 * torch.tensor([np.pi]))
        w_z = px.sample((D,)).squeeze(1)
        w_s = ps.sample((D,)).squeeze(1)

        b_z = p1.sample((D,)).squeeze(1).to(device=zz.device)
        b_s = p2.sample((D,)).squeeze(1).to(device=zz.device)

        # import pdb; pdb.set_trace()
        # if linear == 0:
        phi_z = np.sqrt(2 / D) * torch.cos(torch.mm(zz, w_z.t()) + b_z)
        phi_s = np.sqrt(2 / D) * torch.cos(torch.mm(ss, w_s.t()) + b_s)

        phi_zm = phi_z - torch.mean(phi_z, dim=0)
        phi_sm = phi_s - torch.mean(phi_s, dim=0)

        m_z = phi_zm.shape[1]
        m_s = phi_sm.shape[1]

        C_zz = torch.mm(phi_zm.t(), phi_zm) / n
        C_ss = torch.mm(phi_sm.t(), phi_sm) / n

        C_zs = torch.mm(phi_zm.t(), phi_sm) / n
        A_1 = torch.cat(
            (torch.zeros(m_z, m_z, dtype=C_zs.dtype).to(device=zz.device), torch.mm(torch.linalg.inv(C_zz + lam * torch.eye(m_z, dtype=C_zz.dtype).to(device=zz.device)), C_zs)), dim=1)
        # A_1 = torch.cat((torch.zeros(D, D).cuda(), torch.eye(D).cuda()), dim=1)
        A_2 = torch.cat(
            (torch.mm(torch.linalg.inv(C_ss + lam * torch.eye(m_s, dtype=C_zs.dtype).to(device=zz.device)), C_zs.t()), torch.zeros(m_s, m_s, dtype=C_zs.dtype).to(device=zz.device)),
            dim=1)
        # A_2 = torch.cat((torch.eye(D).cuda(), torch.zeros(D, D).cuda()), dim=1)
        A = torch.cat((A_1, A_2), dim=0)

        # RDC = torch.sqrt(torch.max(torch.abs(torch.linalg.eig(A)[0])))
        kcc = torch.max(torch.real(torch.linalg.eig(A)[0]))

        ########################### dep ##########################################################
        H_z = torch.mm(zz, zz.t())
        # import pdb; pdb.set_trace()
        if self.opts.kernel_labels == 'yes' or self.label == False:
            dep = torch.sqrt(max(torch.trace(torch.mm(K_sm, H_z)) / (n**2), 0))
            # dep = torch.trace(torch.mm(K_sm, H_z)) / torch.trace(torch.mm(K_sm, K_sm))
            # dep = torch.trace(torch.mm(K_sm, H_z)) / torch.sqrt(torch.trace(torch.mm(H_z, H_z)) * torch.trace(torch.mm(K_sm, K_sm)))
        else:
            KL_s = torch.mm(ss, ss.t())
            KL_sm = torch.mm(H, torch.mm(KL_s, H))
            dep = torch.sqrt(torch.trace(torch.mm(KL_sm, H_z)) / (n ** 2))
            # dep = torch.trace(torch.mm(KL_sm, H_z)) / torch.trace(torch.mm(KL_sm, KL_sm))
            # dep = torch.trace(torch.mm(KL_sm, H_z)) / torch.sqrt(torch.trace(torch.mm(H_z, H_z)) * torch.trace(torch.mm(KL_sm, KL_sm)))

        return (kcc**2), dep
        # return hsic


def GaussianKernel(x, s, sigma):
        n_x = x.shape[0]
        n_s = s.shape[0]

        x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
        s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

        ones_x = torch.ones([1, n_x], dtype=x.dtype).to(device=x.device)
        ones_s = torch.ones([1, n_s], dtype=s.dtype).to(device=x.device)

        kernel = torch.exp(
            (-torch.mm(torch.t(x_norm), ones_s) -
             torch.mm(torch.t(ones_x), s_norm) + 2 * torch.mm(x, torch.t(s)))
            / (2 * sigma ** 2))

        return kernel
