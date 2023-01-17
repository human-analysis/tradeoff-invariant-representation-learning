# hsic.py

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

__all__ = ['GaussianKernelndRFFClassification']


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


class GaussianKernelndRFFClassification(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()
        self.data_flag = None
        self.save_hyperparameters()
        self.opts = opts
        self.rbfsigma = getattr(models, opts.gaussian_sigma)()


        ################################### Kernel Encoder ####################################
        self.x = dataloader.x
        self.y = dataloader.y_onehot
        self.y_int = dataloader.y_int
        self.s = dataloader.s

        self.label = dataloader.label

        #################################### finding sigma #######################################
        #
        opts.sigma_x = opts.sigma_x_ratio * self.rbfsigma(self.x, min(self.x.shape[0], 8000))
        if opts.kernel_type_y == 'GaussianKernel' or opts.kernel_type_y == 'LaplacianKernel':
            opts.sigma_y = opts.sigma_y_ratio * self.rbfsigma(self.y, min(self.y.shape[0], 8000))
        if opts.kernel_type_s == 'GaussianKernel' or opts.kernel_type_s == 'LaplacianKernel':
            opts.sigma_s = opts.sigma_s_ratio *self.rbfsigma(self.s, min(self.s.shape[0], 8000))

        self.kernelx = getattr(models, opts.kernel_type)(sigma=opts.sigma_x)
        self.kernely = getattr(models, opts.kernel_type)(sigma=opts.sigma_y)
        self.kernels = getattr(models, opts.kernel_type)(sigma=opts.sigma_s)
        # print(sigmax, sigmay, sigmas)
        # import pdb;
        # pdb.set_trace()

        #########################################################################################

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

        # import pdb; pdb.set_trace()
        phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(self.x, self.w_x.t()) + self.b_x)

        # if opts.kernel_labels == 'yes':
        #     phi_y = np.sqrt(2 / self.D) * torch.cos(torch.mm(self.y, self.w_y.t()) + self.b_y)
        # else:
        #     phi_y = self.y
        # phi_y = self.y / torch.norm(self.y, dim=1)[:, None]

        # phi_s = np.sqrt(2 / self.D) * torch.cos(torch.mm(self.s, self.w_s.t()) + self.b_s)

        # K = torch.mm(phi_s, phi_s.t())

        n = self.x.shape[0]

        Ones = torch.ones(n, 1) / np.sqrt(n)

        # import pdb; pdb.set_trace()
        # L_x = torch.mm(H, phi_x)
        if self.opts.centering == True:
            L_x = phi_x.double() - torch.mm(Ones, torch.mm(Ones.t(), phi_x)).double()
            L_x_c = L_x.doubel()
        else:
            L_x = phi_x.double()
            L_x_c = phi_x.double() - torch.mm(Ones, torch.mm(Ones.t(), phi_x)).double()
        # K = torch.mm(phi_x, phi_x.t())
        # print(torch.linalg.matrix_rank(phi_x))
        # import pdb; pdb.set_trace()
        # L_y = torch.mm(H, phi_y)
        # L_y = phi_y - torch.mm(Ones, torch.mm(Ones.t(), phi_y))
        # L_s = torch.mm(H, phi_s)
        # L_s = phi_s - torch.mm(Ones, torch.mm(Ones.t(), phi_s))
        # import pdb;
        # pdb.set_trace()


        ######################### Full kernel ###############################################

        if opts.kernel_labels == 'yes':
            K_y = self.kernely(self.y, self.y)
        else:
            K_y = torch.mm(self.y, self.y.t())

        if opts.kernel_semantic == 'yes':
            K_s = self.kernels(self.s, self.s)
        else:
            K_s = torch.mm(self.s, self.s.t())
        # print(torch.linalg.matrix_rank(K_y))
        H = torch.eye(n) - torch.ones(n) / n
        K_s = torch.mm(torch.mm(H, K_s), H).double()
        K_y = torch.mm(torch.mm(H, K_y), H).double()

        B_y = torch.mm(L_x.t(), K_y)
        B_y = torch.mm(B_y, L_x)

        B_s = torch.mm(L_x.t(), K_s)
        B_s = torch.mm(B_s, L_x)

        B = B_y / (torch.linalg.norm(B_y, 2)) - opts.tau / (1 - opts.tau) * B_s / (torch.linalg.norm(B_s, 2))
        # B = B_y / (torch.linalg.norm(B_y, 2)) - opts.tau / (1 - opts.tau) * B_s
        # B = B_y - opts.tau / (1 - opts.tau) * B_s / (torch.linalg.norm(B_s, 2))
        # B = B_y - opts.tau / (1 - opts.tau) * B_s

        # A0 = (1 - opts.lam) * torch.mm(L_x.t(), L_x) / n + opts.lam * torch.eye(self.D)
        A = torch.mm(L_x_c.t(), L_x_c)
        # A /= torch.norm(A)
        A /= n
        A = (1 - opts.lam) * A + opts.lam * torch.eye(self.D)
        A = (A + A.t()) / 2
        B = (B + B.t()) / 2
        # B = B.double()
        eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(A.double()), B.double()))
        # eigs0, V0 = torch.linalg.eig(torch.mm(torch.linalg.inv(A.double()), -B.double()))
        # eigs, V = torch.linalg.eig(B.double())
        eigs = torch.real(eigs)
        # eigs0 = torch.real(-eigs0)
        V = torch.real(V)
        ############################################################################################

        sorted, indeces = torch.sort(eigs, descending=True)
        # sorted0, indeces0 = torch.sort(eigs0, descending=True)


        self.eigs = sorted[0:opts.r]

        #########################################################
        # print((torch.norm(sorted[0:12])**2/torch.norm(sorted[0:15])**2).numpy().astype(np.float128))

        # for k in range(15):
        #     if (torch.norm(sorted[0:k])**2/torch.norm(sorted[0:15])**2).numpy().astype(np.float128) > 0.999:
        #         r = k+1
        #         break
        r0 = self.y.shape[1] - 1
        if opts.r_adaptive == 'yes':
            if opts.kernel_labels == 'no' and opts.kernel_semantic == 'no':
                r = r0
            elif opts.tau == 0:
                r = r0
            else:
                r1 = min((sorted > 0).sum(), r0)

            #################### Energy Thresholding ############
                if r1 > 0:
                    for k in range(1, r1+1):
                        if torch.linalg.norm(sorted[0:k])**2 / torch.linalg.norm(sorted[0:r1])**2 >= opts.pca_energy:
                            r = k
                            break
                else:
                    r = 0
            ######################################################
            if opts.tau >= 0.99999:
                r = 0
        else:
            r = r0
        # print(sorted[0:16])
        # import pdb; pdb.set_trace()

        V1 = V[:, indeces[0:opts.r]]
        U = V1.float()
        U[:, r:opts.r] = 0
        # U1 = U[:, self.eigs > 0]

        self.r = r
        # self.theta = torch.mm(torch.pinverse(L_x).t(), U1)

        # self.theta = np.sqrt(n) * U
        self.theta = 15 * U
        # import pdb; pdb.set_trace()
        # self.theta = U * 0 + 1e-16
        ################################# ridge regression ###################################
        # z = torch.mm(torch.mm(K_x, H), self.theta)
        L_x = L_x.float()
        z = torch.mm(L_x, self.theta)
        # import pdb; pdb.set_trace()
        # H = torch.eye(n) - torch.ones(n) / n
        # import pdb;
        # pdb.set_trace()
        # m = z.shape[1]
        # z_c = torch.mm(H, z)
        # y_c = torch.mm(H, self.y)
        self.y_m = torch.mean(self.y, axis=0)
        self.z_m = torch.mean(z, axis=0)


        ####################################################################################################
        self.model = getattr(models, opts.model_type)(**opts.model_options)

        self.criterion = {}
        self.criterion['trn_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)

        self.acc_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)


        if opts.fairness_type is not None:
            self.fairness_train = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            # self.fairness_train1 = getattr(metrics, opts.fairness_type1)(**opts.fairness_options)
            self.fairness_train_t = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            # self.fairness_train1_t = getattr(metrics, opts.fairness_type1)(**opts.fairness_options)
            self.fairness_val = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            # self.fairness_val1 = getattr(metrics, opts.fairness_type1)(**opts.fairness_options)
            self.fairness_val_t = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            # self.fairness_val1_t = getattr(metrics, opts.fairness_type1)(**opts.fairness_options)
            self.fairness_test = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            # self.fairness_test1 = getattr(metrics, opts.fairness_type1)(**opts.fairness_options)
            self.fairness_test_t = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            # self.fairness_test1_t = getattr(metrics, opts.fairness_type1)(**opts.fairness_options)
        else:
            self.fairness_val = None
            self.fairness = None

        ################################################# Data Saving ########################################
        self.data = {}
        self.data['train'] = {'x':[], 'y':[], 's':[]}
        self.data['val'] = {'x':[], 'y':[], 's':[]}
        self.data['test'] = {'x':[], 'y':[], 's':[]}

    def training_step(self, batch, batch_idx):
        x, y, s, label = batch

        self.theta = self.theta.to(device=x.device)
        self.w_x = self.w_x.to(device=x.device)
        self.b_x = self.b_x.to(device=x.device)
        n = x.shape[0]

        phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x, self.w_x.t()) + self.b_x)

        if self.opts.centering == True:
            H = torch.eye(n) - torch.ones(n) / n
            H = H.to(device=x.device)
            L_x = torch.mm(H, phi_x)
        else:
            L_x = phi_x

        z = torch.mm(L_x, self.theta)
        if z.norm() == 0:
            # import pdb;
            # pdb.set_trace()
            z[:, 0] = torch.rand_like(z[:, 0], device=z.device)
            # z = torch.rand_like(z, device=z.device)
        # import pdb; pdb.set_trace()
        # import pdb;
        # pdb.set_trace()
        out = self.model(z)

        ### continuous y ####
        # loss = self.criterion['trn_loss'](out, y)

        ### discrete y #####
        loss = self.criterion['trn_loss'](out, label.long())
        self.log('train_tgt_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        acc = self.acc_trn(out, label.long())
        self.log('train_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        #############################################
        # if (self.current_epoch+1) % self.opts.check_val_every_n_epochs == 0 or self.current_epoch == 0:
        if (self.current_epoch+1) % self.opts.check_val_every_n_epochs == 0:
            # H = torch.eye(n) - torch.ones(n) / n
            # z_c = torch.mm(H.to(device=x.device), z)
            # out_closed = torch.mm(z, self.model_closed.to(device=x.device)) + self.y_m.to(device=x.device)
            # loss_closed = self.acc_trn(out_closed, y).detach()
            # self.log('train_tgt_loss_closed', loss_closed, on_step=False, on_epoch=True, prog_bar=True)

            dep_s = self.fairness_train(z, s, self.opts, self.rbfsigma)
            dep_y = self.fairness_train_t(z, y, self.opts, self.rbfsigma, label=True)

            ####################### Data Saving #######################
            if self.current_epoch == self.opts.nepochs - 1:
                self.data['train']['x'].append(x)
                self.data['train']['y'].append(y)
                self.data['train']['s'].append(s)


        ##############################################

        output = OrderedDict({
            'loss': loss,
            # 'acc': acc
        })
        return output

    def training_epoch_end(self, outputs):
        # if self.current_epoch % self.opts.check_val_every_n_epochs == 0 or self.current_epoch == 0:
        if (self.current_epoch+1) % self.opts.check_val_every_n_epochs == 0:

            kcc_s, dep_s = self.fairness_train.compute()
            kcc_y, dep_y = self.fairness_train_t.compute()


            self.log('train_kcc_s', kcc_s, on_step=False, on_epoch=True)
            self.log('train_dep_s', dep_s, on_step=False, on_epoch=True)

            self.log('train_kcc_y', kcc_y, on_step=False, on_epoch=True)
            self.log('train_dep_y', dep_y, on_step=False, on_epoch=True)

            self.fairness_train.reset()
            # self.fairness_train1.reset()
            self.fairness_train_t.reset()
            # self.fairness_train1_t.reset()

        else:
            pass

        self.acc_trn.reset()

    def validation_step(self, batch, batch_idx):
        x, y, s, label = batch
        self.theta = self.theta.to(device=x.device)
        self.w_x = self.w_x.to(device=x.device)
        self.b_x = self.b_x.to(device=x.device)
        n = x.shape[0]

        phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x, self.w_x.t()) + self.b_x)

        if self.opts.centering == True:
            H = torch.eye(n) - torch.ones(n) / n
            H = H.to(device=x.device)
            L_x = torch.mm(H, phi_x)
        else:
            L_x = phi_x

        z = torch.mm(L_x, self.theta)
        if z.norm() == 0:
            z[:, 0] = torch.rand_like(z[:, 0], device=z.device)
            # z = torch.rand_like(z, device=z.device)

        out = self.model(z)

        ### continuous y ####
        # loss = self.criterion['val_loss'](out, y)

        ### discrete y #####
        loss = self.criterion['val_loss'](out, label.long())
        self.log('val_tgt_loss', loss, on_step=False, on_epoch=True)

        acc = self.acc_val(out, label.long())
        self.log('val_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        #########################################################################################

        # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_embedding_dim', self.r, on_step=False, on_epoch=True)
        # self.log('val_acc_bayes', acc_bayes, on_step=False, on_epoch=True, prog_bar=True)

        # output = OrderedDict({
        #     'loss': loss
        #     # 'acc': acc
        # })

        if self.fairness_val is not None:
            kcc_s = self.fairness_val(z, s, self.opts, self.rbfsigma)
            kcc_y = self.fairness_val_t(z, y, self.opts, self.rbfsigma, label=True)

        ################### Data Saving ###########################
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['val']['x'].append(x)
            self.data['val']['y'].append(y)
            self.data['val']['s'].append(s)

    def validation_epoch_end(self, outputs):
        kcc_s, dep_s = self.fairness_val.compute()
        kcc_y, dep_y = self.fairness_val_t.compute()


        self.log('val_kcc_s', kcc_s, on_step=False, on_epoch=True)
        self.log('val_dep_s', dep_s, on_step=False, on_epoch=True)

        self.log('val_kcc_y', kcc_y, on_step=False, on_epoch=True)
        self.log('val_dep_y', dep_y, on_step=False, on_epoch=True)

        self.fairness_val.reset()
        # self.fairness_val1.reset()
        self.fairness_val_t.reset()
        # self.fairness_val1_t.reset()

    def test_step(self, batch, batch_idx):
        x, y, s, label = batch
        # x_c = x - self.x_m.to(device=x.device)
        self.theta = self.theta.to(device=x.device)
        self.w_x = self.w_x.to(device=x.device)
        self.b_x = self.b_x.to(device=x.device)
        n = x.shape[0]
        phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x, self.w_x.t()) + self.b_x)

        if self.opts.centering == True:
            H = torch.eye(n) - torch.ones(n) / n
            H = H.to(device=x.device)
            L_x = torch.mm(H, phi_x)
        else:
            L_x = phi_x

        z = torch.mm(L_x, self.theta)
        if z.norm() == 0:
            # z = torch.rand_like(z, device=z.device)
            z[:, 0] = torch.rand_like(z[:, 0], device=z.device)


        ###############################################################################################
        # z = x

        out = self.model(z)
        ### continuous y ####
        # loss = self.criterion['test_loss'](out, y)

        ### discrete y #####
        loss = self.criterion['test_loss'](out, label.long())

        acc = self.acc_tst(out, label.long())

        # import pdb; pdb.set_trace()
        # acc = self.acc_tst(out, y)

        # self.log('test_acc_bayes', acc_bayes, on_step=False, on_epoch=True, prog_bar=True)

        # import pdb; pdb.set_trace()

        output = OrderedDict({
            'loss': loss
            # 'acc': acc
        })


        if self.fairness_test is not None:
            kcc_s = self.fairness_test(z, s, self.opts, self.rbfsigma)
            kcc_y = self.fairness_test_t(z, y, self.opts, self.rbfsigma, label=True)

        #################### Data Saving ##############################
        self.data['test']['x'].append(x)
        self.data['test']['y'].append(y)
        self.data['test']['s'].append(s)

    def test_epoch_end(self, outputs):
        metrics_dict = {}
        kcc_s, dep_s = self.fairness_test.compute()
        kcc_y, dep_y = self.fairness_test_t.compute()

        acc = self.acc_tst.compute()
        loss = self.criterion['test_loss'].compute()
        # test_dep_y = self.nncc['y_test'].compute()

        if self.data_flag is not None:
            if self.data_flag:


                self.log('test_test_kcc_s', kcc_s, on_step=False, on_epoch=True)
                self.log('test_test_dep_s', dep_s, on_step=False, on_epoch=True)

                self.log('test_test_kcc_y', kcc_y, on_step=False, on_epoch=True)
                self.log('test_test_dep_y', dep_y, on_step=False, on_epoch=True)

                metrics_dict['test_test_kcc_s'] = float(kcc_s.cpu().numpy().astype(np.float128))
                metrics_dict['test_test_dep_s'] = float(dep_s.cpu().numpy().astype(np.float128))

                metrics_dict['test_test_kcc_y'] = float(kcc_y.cpu().numpy().astype(np.float128))
                metrics_dict['test_test_dep_y'] = float(dep_y.cpu().numpy().astype(np.float128))


                metrics_dict['test_eigs'] = self.flatten(self.eigs)
                # metrics_dict['test_eigs1'] = self.flatten(self.eigs1)
                # import pdb; pdb.set_trace()

                self.log('test_test_acc', acc, on_step=False, on_epoch=True)
                self.log('test_test_tgt_loss', loss, on_step=False, on_epoch=True)
                self.log('test_embedding_dim', self.r, on_step=False, on_epoch=True)

                metrics_dict['test_acc'] = float(acc.cpu().numpy().astype(np.float128))

                metrics_dict['test_test_tgt_loss'] = float(loss.cpu().numpy().astype(np.float128))

                metrics_dict['test_embedding_dim'] = int(self.r)

                ####################################### Data Saving ######################################
                for split_name, data in self.data.items():
                    # import pdb; pdb.set_trace()
                    x = torch.cat(data['x'], dim=0)
                    y = torch.cat(data['y'], dim=0)
                    s = torch.cat(data['s'], dim=0)
                    #######################################################
                    self.theta = self.theta.to(device=x.device)
                    self.w_x = self.w_x.to(device=x.device)
                    self.b_x = self.b_x.to(device=x.device)
                    n = x.shape[0]

                    phi_x = np.sqrt(2 / self.D) * torch.cos(torch.mm(x, self.w_x.t()) + self.b_x)
                    # import pdb; pdb.set_trace()

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
            else:

                self.log('test_train_kcc_s', kcc_s, on_step=False, on_epoch=True)
                self.log('test_train_dep_s', dep_s, on_step=False, on_epoch=True)

                self.log('test_train_kcc_y', kcc_y, on_step=False, on_epoch=True)
                self.log('test_train_dep_y', dep_y, on_step=False, on_epoch=True)


                metrics_dict['test_train_kcc_s'] = float(kcc_s.cpu().numpy().astype(np.float128))
                metrics_dict['test_train_dep_s'] = float(dep_s.cpu().numpy().astype(np.float128))

                metrics_dict['test_train_kcc_y'] = float(kcc_y.cpu().numpy().astype(np.float128))
                metrics_dict['test_train_dep_y'] = float(dep_y.cpu().numpy().astype(np.float128))

                self.log('test_train_acc', acc, on_step=False, on_epoch=True)
                self.log('test_train_tgt_loss', loss, on_step=False, on_epoch=True)

                metrics_dict['test_train_acc'] = float(acc.cpu().numpy().astype(np.float128))

                metrics_dict['test_train_tgt_loss'] = float(loss.cpu().numpy().astype(np.float128))


        self.fairness_test.reset()
        # self.fairness_test1.reset()
        self.fairness_test_t.reset()
        # self.fairness_test1_t.reset()
        # self.criterion['test_loss'].reset()


        # self.acc_tst.reset()
        self.to_txt(**metrics_dict)

    def flatten(self, x):
        x = x.numpy().reshape(-1)
        out = '['

        for i, el in enumerate(x):
            out += f'{el}'
            if i < len(x) -1:
                out += ', '

        out += ']'
        return out


    def to_txt(self, **kwargs):
        random_seed = self.opts.manual_seed
        tau = self.opts.tau
        txt = f'{{"random_seed": {random_seed}, "tau": {tau}'
        # txt = f'{{ "tau": {tau}'

        for key, value in kwargs.items():
            txt += f', "{key}": {value}'

        txt += '}\n'

        # print(txt)
        file_dir = self.opts.result_path
        # file_name = self.opts.results_txt_file
        file_name = 'kernel.txt'

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
