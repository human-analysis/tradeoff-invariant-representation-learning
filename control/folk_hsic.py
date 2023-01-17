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

__all__ = ['FolkHSIC']


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


class FolkHSIC(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        # self.data_flag = None
        self.save_hyperparameters()
        self.opts = opts

        self.opts.rbfsigma = getattr(models, opts.gaussian_sigma)()

        ##########################################################################################

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
        opts.fairness_options1['num_s_classes'] = int(self.opts.num_sensitive_attrs)
        opts.fairness_options1['num_sensitive_att'] = 1

        self.fairness1_val = getattr(metrics, opts.fairness_type1)(**opts.fairness_options1)
        self.fairness1_test = getattr(metrics, opts.fairness_type1)(**opts.fairness_options1)

        self.data = {}
        self.data['train'] = {'x': [], 'y': [], 's': []}
        self.data['val'] = {'x': [], 'y': [], 's': []}
        self.data['test'] = {'x': [], 'y': [], 's': []}

        self.metrics_dict = {}

    def hsic_loss(self, zz, ss, n):
        torch.set_default_tensor_type(torch.DoubleTensor)

        zz, ss = zz.double(), ss.double()
        sigma_z = self.opts.rbfsigma(zz, zz.shape[0])
        sigma_s = self.opts.rbfsigma(ss, ss.shape[0])

        #################################################3
        H = torch.eye(n) - torch.ones(n) / n
        H = H.to(device=ss.device)

        K_s = GaussianKernel(ss.to(dtype=H.dtype), ss.to(dtype=H.dtype), sigma_s)
        K_sm = torch.mm(H, torch.mm(K_s, H))
        K_z = GaussianKernel(zz.to(dtype=H.dtype), zz.to(dtype=H.dtype), sigma_z)
        K_zm = torch.mm(H, torch.mm(K_z, H))

        # hsic = torch.trace(torch.mm(K_zm, K_sm)) / ((n-1)**2)
        hsic = torch.trace(torch.mm(K_zm, K_sm)) / torch.sqrt(
            torch.trace(torch.mm(K_sm, K_sm)) * torch.trace(torch.mm(K_zm, K_zm)))

        torch.set_default_tensor_type(torch.FloatTensor)

        return hsic

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        if self.opts.age_remove == 'yes':
            x = x[:, 1:]

        # s_onehot = self.format_s_onehot(s)
        y_onehot = self.format_y_onehot(y)

        n = x.shape[0]
        z, out = self.model(x)

        loss_target = self.criterion['trn_loss'](out, y)
        acc = self.acc_trn(out, y.int())

        self.log('train_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_tgt_loss', loss_target, on_step=False, on_epoch=True, prog_bar=True)

        ########################### HSIC Kernel #########################################################

        hsic = self.hsic_loss(z, s, n)

        self.log('train_hsic_loss', hsic, on_step=False, on_epoch=True, prog_bar=True)

        #####################################################################################
        loss = (1 - self.opts.tau) * loss_target + self.opts.tau * hsic

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

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

    def validation_step(self, batch, batch_idx):
        x, y, s = batch
        if self.opts.age_remove == 'yes':
            x = x[:, 1:]
        n = x.shape[0]

        # s_onehot = self.format_s_onehot(s)
        y_onehot = self.format_y_onehot(y)

        z, out = self.model(x)
        loss_target = self.criterion['val_loss'](out, y)
        acc = self.acc_val(out, y.int())

        ########################### HSIC Kernel #########################################################

        hsic = self.hsic_loss(z, s, n)
        ########################################################################################################

        self.log('val_loss', loss_target, on_step=False, on_epoch=True)  # for checkpoint
        self.log('val_tgt_loss', loss_target, on_step=False, on_epoch=True)
        self.log('val_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hsic_loss', hsic, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_embedding_dim', self.opts.r, on_step=False, on_epoch=True)

        DPV = self.fairness1_val(out, s)

        # Appending Z   
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['val']['x'].append(x)
            self.data['val']['y'].append(y)
            self.data['val']['s'].append(s)



    def validation_epoch_end(self, outputs):


        ############################## log DPV Metric ##################################
        out, DPV_var, DPV_max = self.fairness1_val.compute()

        self.log(f'val_DPV_var', float(DPV_var.cpu().numpy().astype(np.float128)))
        # self.log(f'val_DPV_max', float(DPV_max.cpu().numpy().astype(np.float128)))
        # self.log(f'val_DPV_max', float(DPV_max))

        self.fairness1_val.reset()

        self.acc_val.reset()
        # self.criterion['val_loss'].reset()

    def test_step(self, batch, batch_idx):
        x, y, s = batch
        if self.opts.age_remove == 'yes':
            x = x[:, 1:]
        n = x.shape[0]

        # s_onehot = self.format_s_onehot(s)
        y_onehot = self.format_y_onehot(y)

        z, out = self.model(x)

        loss = self.criterion['test_loss'](out, y)
        acc = self.acc_tst(out, y.int())

        ########################### HSIC Kernel #########################################################

        hsic = self.hsic_loss(z, s, n)
        ########################################################################################################

        self.log('test_embedding_dim', self.opts.r, on_step=False, on_epoch=True)
        self.log('test_tgt_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hsic_loss', hsic, on_step=False, on_epoch=True, prog_bar=True)


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
            z, _ = self.model(x)

            y = torch.cat(data['y'], dim=0)
            s = torch.cat(data['s'], dim=0)

            np.savetxt(self.opts.out_dir + f'/z_{split_name}.out', z.detach().cpu(), fmt='%10.5f')
            np.savetxt(self.opts.out_dir + f'/y_{split_name}.out', y.detach().cpu(), fmt='%10.5f')
            np.savetxt(self.opts.out_dir + f'/s_{split_name}.out', s.detach().cpu(), fmt='%10.5f')

        ############################## log DPV Metric #########################################
        out, DPV_var, DPV_max = self.fairness1_test.compute()

        self.log('test_DPV_var', float(DPV_var.cpu().numpy().astype(np.float128)))
        metrics_dict['test_DPV_var'] = float(DPV_var.cpu().numpy().astype(np.float128))

        self.log(f'test_DPV_max', float(DPV_max))
        metrics_dict['test_DPV_max'] = float(DPV_max.cpu().numpy().astype(np.float128))

        self.fairness1_test.reset()

        metrics_dict['test_embedding_dim'] = int(self.opts.r)
        metrics_dict['test_tgt_loss'] = float(self.criterion['test_loss'].compute().cpu().numpy().astype(np.float128))
        metrics_dict['test_tgt_acc'] = float(self.acc_tst.compute().cpu().numpy().astype(np.float128))

        # self.criterion['test_loss'].reset()
        self.acc_tst.reset()


        self.to_txt(**metrics_dict)

    def format_y_onehot(self, y):
        y_onehot = torch.zeros(y.size(0), self.opts.model_options['nclasses'], device=y.device).scatter_(1, y.unsqueeze(
            1).type(torch.int64), 1)
        return y_onehot

    def format_s_onehot(self, s):
        # int -> one-hot
        s_onehot = torch.zeros(s.size(0), self.opts.num_sensitive_attrs, device=s.device).scatter_(1, s.unsqueeze(
            1).long(), 1)

        return s_onehot.cuda()

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