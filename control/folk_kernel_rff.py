import torch
import time
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
import os

import hal.models as models
import hal.losses as losses
import hal.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np
import scipy.io as sio
from scipy.linalg import eigh
import torch.nn.functional as F
from sklearn.manifold import TSNE
import scipy.stats as ss

__all__ = ['FolkKernelRFF']


def GaussianKernel(x, s, sigma):
    n_x = x.shape[0]
    n_s = s.shape[0]

    x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
    s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

    ones_x = torch.ones([1, n_x]).to(device=x.device)
    ones_s = torch.ones([1, n_s]).to(device=x.device)

    kernel = torch.exp(
        (-torch.mm(torch.t(x_norm), ones_s) -
         torch.mm(torch.t(ones_x), s_norm) + 2 * torch.mm(x, torch.t(s)))
        / (2 * sigma ** 2))

    return kernel


class FolkKernelRFF(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()
        # self.data_flag = None
        self.save_hyperparameters()
        self.opts = opts
        self.kernelx = getattr(models, opts.kernel_type)(sigma=opts.sigma_x)
        self.kernely = getattr(models, opts.kernel_type)(sigma=opts.sigma_y)
        self.kernels = getattr(models, opts.kernel_type)(sigma=opts.sigma_s)
        self.rbfsigma = getattr(models, opts.gaussian_sigma)()

        # import pdb; pdb.set_trace()

        torch.set_default_tensor_type(torch.DoubleTensor)
        ################################### Kernel Encoder ####################################
        data_dict = dataloader.data
        x = data_dict['train']['x']
        y = data_dict['train']['y']
        s = data_dict['train']['s']

        # import pdb; pdb.set_trace()

        # assert y.max() == 1 and y.min() == 0, f"y is not binary:: max(y)={y.max()} and min(y)={y.min()}"

        self.x = x.double()  # .float()#.cuda()
        if opts.age_remove == 'yes':
            self.x = self.x[:, 1:]
        # self.y = y.unsqueeze(1).float()
        self.y = self.format_y_onehot(y)  # .float()
        # s_onehot = self.format_s_onehot(s)

        # self.s = s.unsqueeze(1)
        self.s = s
        # print((y==0).sum()/len(y))
        # import pdb; pdb.set_trace()
        ######################################################################################################

        #################################### finding sigma #######################################
        # opts.sigma_x = self.rbfsigma(self.x, min(self.x.shape[0], 15000))
        # opts.sigma_y = self.rbfsigma(self.y, min(self.y.shape[0], 15000))
        # opts.sigma_s = self.rbfsigma(self.s, min(self.s.shape[0], 15000))
        # print(f'sigma_x = {opts.sigma_x}\nsigma_y = {opts.sigma_y}\nsigma_s = {opts.sigma_s}')
        # import pdb; pdb.set_trace()
        # exit()
        #########################################################################################

        print('Preparing the kernel ...')
        self.D = opts.drff

        # p(w) = exp(-sigma^2||w||^2 / 2)
        torch.manual_seed(0)
        Sigma_x = 1 / (opts.sigma_x ** 2) * torch.diag(torch.ones(self.x.shape[1]))
        torch.manual_seed(1)
        Sigma_y = 1 / (opts.sigma_y ** 2) * torch.diag(torch.ones(self.y.shape[1]))
        torch.manual_seed(2)
        Sigma_s = 1 / (opts.sigma_s ** 2) * torch.diag(torch.ones(self.s.shape[1]))
        torch.manual_seed(3)
        Ones_x = torch.zeros(1, self.x.shape[1])
        torch.manual_seed(4)
        Ones_y = torch.zeros(1, self.y.shape[1])
        torch.manual_seed(5)
        Ones_s = torch.zeros(1, self.s.shape[1])

        px = torch.distributions.MultivariateNormal(Ones_x, Sigma_x)
        py = torch.distributions.MultivariateNormal(Ones_y, Sigma_y)
        ps = torch.distributions.MultivariateNormal(Ones_s, Sigma_s)
        p1 = torch.distributions.uniform.Uniform(torch.tensor([0.0]), 2 * torch.tensor([np.pi]))
        p2 = torch.distributions.uniform.Uniform(torch.tensor([0.0]), 2 * torch.tensor([np.pi]))
        p3 = torch.distributions.uniform.Uniform(torch.tensor([0.0]), 2 * torch.tensor([np.pi]))

        self.w_x = px.sample((self.D,)).squeeze(1)
        self.w_y = py.sample((self.D,)).squeeze(1)
        self.w_s = ps.sample((self.D,)).squeeze(1)

        self.b_x = p1.sample((self.D,)).squeeze(1)
        self.b_y = p2.sample((self.D,)).squeeze(1)
        self.b_s = p3.sample((self.D,)).squeeze(1)

        if opts.kernel_data == 'yes':
            phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(self.x, self.w_x.t()) + self.b_x)
        else:
            phi_x = self.x

        if opts.kernel_labels == 'yes':
            phi_y = np.sqrt(2 / self.D) * torch.cos(torch.mm(self.y, self.w_y.t()) + self.b_y)
        else:
            phi_y = self.y

        if opts.kernel_semantic == 'yes':
            phi_s = np.sqrt(2 / self.D) * torch.cos(torch.mm(self.s, self.w_s.t()) + self.b_s)
        else:
            phi_s = self.s

        print('Completed part 1')

        # K = torch.mm(phi_s, phi_s.t())

        n = self.x.shape[0]

        Ones = torch.ones(n, 1) / np.sqrt(n)

        # import pdb; pdb.set_trace()
        # L_x = torch.mm(H, phi_x)
        if self.opts.centering == True:
            L_x = phi_x - torch.mm(Ones, torch.mm(Ones.t(), phi_x))
            L_x_c = L_x
        else:
            L_x = phi_x
            L_x_c = phi_x - torch.mm(Ones, torch.mm(Ones.t(), phi_x))
        # K = torch.mm(phi_x, phi_x.t())
        # print(torch.linalg.matrix_rank(phi_x))
        # import pdb; pdb.set_trace()
        # L_y = torch.mm(H, phi_y)
        L_y = phi_y - torch.mm(Ones, torch.mm(Ones.t(), phi_y))
        # L_s = torch.mm(H, phi_s)
        L_s = phi_s - torch.mm(Ones, torch.mm(Ones.t(), phi_s))
        # import pdb;
        # pdb.set_trace()
        print('Completed part 2')
        ##############################################################################

        B_y = torch.mm(L_x.t(), L_y)
        B_y = torch.mm(B_y, B_y.t())

        B_s = torch.mm(L_x.t(), L_s)
        B_s = torch.mm(B_s, B_s.t())

        # B = (1 - opts.tau) * B_y / torch.norm(B_y) - opts.tau * B_s / torch.norm(B_s)
        B = B_y / (torch.linalg.norm(B_y, 2)) - opts.tau / (1 - opts.tau) * B_s / (torch.linalg.norm(B_s, 2))

        # A0 = (1 - opts.lam) * torch.mm(L_x.t(), L_x) / n + opts.lam * torch.eye(self.D)
        A = torch.mm(L_x_c.t(), L_x_c)
        # A /= torch.norm(A)
        A /= n
        A += opts.lam * torch.eye(A.shape[0])

        A = (A + A.t()) / 2

        B = (B + B.t()) / 2

        # eigs, V = torch.lobpcg(B, B=A, k=opts.r, method='ortho', largest=True)
        eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(A), B))
        eigs = torch.real(eigs)
        V = torch.real(V)
        sorted, indeces = torch.sort(eigs, descending=True)

        self.eigs = sorted[0:opts.r]
        U = V[:, indeces[0:opts.r]]
        # U1 = U0[:, eigs0 > 0]
        # U1 = U

        ###############################################################################3
        # r0 = self.y.shape[1] - 1
        r0 = opts.r
        if opts.tau == 0:
            r = r0
        else:
            r1 = min((sorted > 0).sum(), r0)

            #################### Energy Thresholding ############
            if r1 > 0:
                for k in range(1, r1 + 1):
                    if torch.linalg.norm(sorted[0:k]) ** 2 / torch.linalg.norm(sorted[0:r1]) ** 2 >= opts.pca_energy:
                        r = k
                        break
            else:
                r = 0
        ######################################################
        if opts.tau >= 0.99999:
            r = 0
        ###############################################################################

        # import pdb; pdb.set_trace()
        # r = min((sorted > 0).sum(), opts.r)
        U[:, r:opts.r] = 0
        self.r = r
        # self.theta = torch.mm(torch.pinverse(L_x).t(), U1)

        # self.theta = np.sqrt(n) * U
        self.theta = 30 * U
        torch.set_default_tensor_type(torch.FloatTensor)
        # import pdb; pdb.set_trace()
        # self.theta = U * 0 + 1e-16
        print('Completed part 3')

        print('Preparing the kernel is done!')
        ####### Kernel ########
        # sigma_z = self.rbfsigma(z, z.shape[0])
        # K_z = GaussianKernel(z, z, sigma_z)
        # I = torch.eye(n).to(device=K_z.device)
        # K = K_z + 0.01 * I
        # self.model_kernel = torch.mm(torch.linalg.inv(K), self.y)
        # self.z = z

        ####################################################################################################
        # import pdb; pdb.set_trace()
        self.model = getattr(models, opts.model_type)(**opts.model_options)

        self.criterion = {}
        self.criterion['trn_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)

        self.acc_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)

        self.acc_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_tst_kernel = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)

        self.acc_tst_max = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)

        #################################################################################
        #                                   DPV Metric                                  #
        #################################################################################
        # DPV Metric
        opts.fairness_options1['num_y_classes'] = int(opts.model_options['nclasses'])
        # opts.fairness_options1['num_s_classes'] = int(opts.model_adv_options['nclasses'])
        opts.fairness_options1['num_s_classes'] = int(opts.num_sensitive_attrs)
        opts.fairness_options1['num_sensitive_att'] = 1

        self.fairness1_val = getattr(metrics, opts.fairness_type1)(**opts.fairness_options1)
        self.fairness1_test = getattr(metrics, opts.fairness_type1)(**opts.fairness_options1)

        self.data = {}
        self.data['train'] = {'x': [], 'y': [], 's': []}
        self.data['val'] = {'x': [], 'y': [], 's': []}
        self.data['test'] = {'x': [], 'y': [], 's': []}

        self.metrics_dict = {}

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        if self.opts.age_remove == 'yes':
            x = x[:, 1:]

        # s_onehot = self.format_s_onehot(s)
        # y_onehot = self.format_y_onehot(y)

        self.theta = self.theta.to(device=x.device)
        self.w_x = self.w_x.to(device=x.device)
        self.b_x = self.b_x.to(device=x.device)
        n = x.shape[0]

        if self.opts.kernel_data == 'yes':
            phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x.to(dtype=self.w_x.dtype), self.w_x.t()) + self.b_x)
        else:
            phi_x = x.to(dtype=self.w_x.dtype)

        if self.opts.centering == True:
            H = torch.eye(n) - torch.ones(n) / n
            H = H.to(device=x.device, dtype=self.w_x.dtype)
            L_x = torch.mm(H, phi_x)
        else:
            L_x = phi_x

        # import pdb;
        # pdb.set_trace()
        z = torch.mm(L_x, self.theta).float()
        if z.norm() == 0:
            z[:, 0] = torch.rand_like(z[:, 0], device=z.device)
        out = self.model(z)
        # import pdb; pdb.set_trace()
        loss = self.criterion['trn_loss'](out, y)
        acc = self.acc_trn(out, y.int())

        self.log('train_tgt_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # Appending Z
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['train']['x'].append(x)
            self.data['train']['y'].append(y)
            self.data['train']['s'].append(s)


        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output

    def training_epoch_end(self, outputs):

        self.acc_trn.reset()
        # self.criterion['trn_loss'].reset()
        # print('Train Done!')

    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        if self.opts.age_remove == 'yes':
            x = x[:, 1:]

        # s_onehot = self.format_s_onehot(s)
        # y_onehot = self.format_y_onehot(y)

        self.theta = self.theta.to(device=x.device)
        self.w_x = self.w_x.to(device=x.device)
        self.b_x = self.b_x.to(device=x.device)
        n = x.shape[0]

        # import pdb; pdb.set_trace()
        if self.opts.kernel_data == 'yes':
            phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x.to(dtype=self.w_x.dtype), self.w_x.t()) + self.b_x)
        else:
            phi_x = x.to(dtype=self.w_x.dtype)

        if self.opts.centering == True:
            H = torch.eye(n) - torch.ones(n) / n
            H = H.to(device=x.device, dtype=self.w_x.dtype)
            L_x = torch.mm(H, phi_x)
        else:
            L_x = phi_x

        z = torch.mm(L_x, self.theta).float()
        if z.norm() == 0:
            z[:, 0] = torch.rand_like(z[:, 0], device=z.device)


        out = self.model(z)
        loss = self.criterion['val_loss'](out, y)
        # import pdb; pdb.set_trace()
        acc = self.acc_val(out, y.int())
        self.log('val_loss', loss, on_step=False, on_epoch=True)  # for checkpoint
        self.log('val_tgt_loss', loss, on_step=False, on_epoch=True)
        self.log('val_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_embedding_dim', self.r, on_step=False, on_epoch=True)
        # self.log('val_acc_bayes', acc_bayes, on_step=False, on_epoch=True, prog_bar=True)

        DPV = self.fairness1_val(out, s)

        # Appending Z
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['val']['x'].append(x)
            self.data['val']['y'].append(y)
            self.data['val']['s'].append(s)

    def validation_epoch_end(self, outputs):

        ############################## log DPV Metric ##################################
        out, DPV_var, DPV_max = self.fairness1_val.compute()
        # import pdb; pdb.set_trace()
        self.log(f'val_DPV_var', float(DPV_var.cpu().numpy().astype(np.float128)))
        # self.log(f'val_DPV_max', float(DPV_max.cpu().numpy().astype(np.float128)))
        # self.log(f'val_DPV_max', float(DPV_max))

        self.fairness1_val.reset()

        self.acc_val.reset()
        # self.criterion['val_loss'].reset()
        # print('Val Done!')

    def test_step(self, batch, batch_idx):
        x, y, s = batch
        if self.opts.age_remove == 'yes':
            x = x[:, 1:]

        # s_onehot = self.format_s_onehot(s)
        y_onehot = self.format_y_onehot(y)

        # x_c = x - self.x_m.to(device=x.device)
        self.theta = self.theta.to(device=x.device)
        self.w_x = self.w_x.to(device=x.device)
        self.b_x = self.b_x.to(device=x.device)
        n = x.shape[0]
        if self.opts.kernel_data == 'yes':
            phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x.to(dtype=self.w_x.dtype), self.w_x.t()) + self.b_x)
        else:
            phi_x = x.to(dtype=self.w_x.dtype)

        if self.opts.centering == True:
            H = torch.eye(n) - torch.ones(n) / n
            H = H.to(device=x.device, dtype=self.w_x.dtype)
            L_x = torch.mm(H, phi_x)
        else:
            L_x = phi_x

        z = torch.mm(L_x, self.theta).float()
        if z.norm() == 0:
            z[:, 0] = torch.rand_like(z[:, 0], device=z.device)

        # alpha = torch.rand(1)

        out = self.model(z)
        loss = self.criterion['test_loss'](out, y)
        acc = self.acc_tst(out, y.int())

        # self.log('test_utility', max(0, 1-loss/self.loss_max), on_step=False, on_epoch=True)

        self.log('test_embedding_dim', self.r, on_step=False, on_epoch=True)
        self.log('test_tgt_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # import pdb; pdb.set_trace()

        # output = OrderedDict({
        # 'loss': loss,
        # 'acc': acc
        # })

        # Update DPV Metric
        DPV = self.fairness1_test(out, s)

        # Appending Z
        self.data['test']['x'].append(x)
        self.data['test']['y'].append(y)
        self.data['test']['s'].append(s)


    def test_epoch_end(self, outputs):
        metrics_dict = self.metrics_dict

        ################################## Saving Z #########################################
        for split_name, data in self.data.items():
            # import pdb; pdb.set_trace()
            x = torch.cat(data['x'], dim=0)
            #######################################################
            self.theta = self.theta.to(device=x.device)
            self.w_x = self.w_x.to(device=x.device)
            self.b_x = self.b_x.to(device=x.device)
            n = x.shape[0]
            # import pdb; pdb.set_trace()

            if self.opts.kernel_data == 'yes':
                phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x.to(dtype=self.w_x.dtype), self.w_x.t()) + self.b_x)
            else:
                phi_x = x.to(dtype=self.w_x.dtype)

            if self.opts.centering == True:
                H = torch.eye(n) - torch.ones(n) / n
                H = H.to(device=x.device)
                L_x = torch.mm(H, phi_x)
            else:
                L_x = phi_x

            z = torch.mm(L_x, self.theta)
            #######################################################
            # z, _ = self.model(x)

            y = torch.cat(data['y'], dim=0)
            s = torch.cat(data['s'], dim=0)

            np.savetxt(self.opts.out_dir + f'/z_{split_name}.out', z.detach().cpu(), fmt='%10.5f')
            np.savetxt(self.opts.out_dir + f'/y_{split_name}.out', y.detach().cpu(), fmt='%10.5f')
            np.savetxt(self.opts.out_dir + f'/s_{split_name}.out', s.detach().cpu(), fmt='%10.5f')

        ############################## log DPV Metric #########################################
        out, DPV_var, DPV_max = self.fairness1_test.compute()

        self.log('test_DPV_var', float(DPV_var.cpu().numpy().astype(np.float128)))
        metrics_dict['test_DPV_var'] = float(DPV_var.cpu().numpy().astype(np.float128))

        # self.log(f'test_DPV_max', float(DPV_max))
        # metrics_dict['test_DPV_max'] = float(DPV_max.cpu().numpy().astype(np.float128))

        self.fairness1_test.reset()

        metrics_dict['test_embedding_dim'] = int(self.r)
        metrics_dict['test_tgt_loss'] = float(self.criterion['test_loss'].compute().cpu().numpy().astype(np.float128))
        metrics_dict['test_tgt_acc'] = float(self.acc_tst.compute().cpu().numpy().astype(np.float128))

        metrics_dict['test_lam'] = float(self.opts.lam)

        # self.criterion['test_loss'].reset()
        self.acc_tst.reset()
        '''
        self.nncc['s_test'].reset()
        self.nncc['y_test'].reset()
        '''
        self.to_txt(**metrics_dict)

    def format_y_onehot(self, y):
        y_onehot = torch.zeros(y.size(0), self.opts.model_options['nclasses'], device=y.device).scatter_(1, y.unsqueeze(
            1).type(torch.int64), 1)
        return y_onehot

    def format_s_onehot(self, s):
        # int -> one-hot
        s_onehot = torch.zeros(s.size(0), self.opts.num_sensitive_attrs, device=s.device).scatter_(1, s.unsqueeze(
            1).long(), 1)

        return s_onehot

    def flatten(self, x):
        x = x.numpy().reshape(-1)
        out = '['

        for i, el in enumerate(x):
            out += f'{el}'
            if i < len(x) - 1:
                out += ', '

        out += ']'
        return out

    def to_txt(self, **kwargs):
        random_seed = self.opts.manual_seed
        tau = self.opts.tau
        txt = f'{{"random_seed": {random_seed}, "tau": {tau}'

        for key, value in kwargs.items():
            txt += f', "{key}": {value}'

        txt += '}\n'

        file_dir = self.opts.result_path
        file_name = self.opts.results_txt_file

        try:
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        except:
            pass

        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'a') as f:
            f.write(txt)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)
        if self.opts.scheduler_method is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer, **self.opts.scheduler_options
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]