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
# from scipy.linalg import eigh
from sklearn.manifold import TSNE

__all__ = ['GaussianARLClassification']


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


class GaussianARLClassification(pl.LightningModule):
    def __init__(self, opts, dataloader):
        super().__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.data_flag = None
        self.save_hyperparameters()
        self.kernelx = getattr(models, opts.kernel_type)(sigma=opts.sigma_x)
        self.kernely = getattr(models, opts.kernel_type)(sigma=opts.sigma_y)
        self.kernels = getattr(models, opts.kernel_type)(sigma=opts.sigma_s)
        self.rbfsigma = getattr(models, opts.gaussian_sigma)()
        self.opts = opts
        self.r = opts.r

        ####################################################################################################
        self.model = getattr(models, opts.model_type)(**opts.model_options)
        self.model_adv = getattr(models, opts.model_adv_type)(**opts.model_options_adv)
        self.rbfsigma = getattr(models, opts.gaussian_sigma)()

        self.criterion = {}
        self.criterion['tgt_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['adv_loss'] = getattr(losses, opts.adv_loss_type)(**opts.adv_loss_options)
        self.criterion['test_tgt_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_adv_loss'] = getattr(losses, opts.adv_loss_type)(**opts.adv_loss_options)
        self.criterion['val_tgt_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_adv_loss'] = getattr(losses, opts.adv_loss_type)(**opts.adv_loss_options)

        self.loss_tgt_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.loss_adv_trn = getattr(metrics, opts.adv_evaluation_type)(**opts.adv_evaluation_options)

        self.loss_tgt_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.loss_adv_val = getattr(metrics, opts.adv_evaluation_type)(**opts.adv_evaluation_options)

        self.loss_tgt_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.loss_adv_tst = getattr(metrics, opts.adv_evaluation_type)(**opts.adv_evaluation_options)

        self.acc_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)


        ########################################################################################

        if opts.fairness_type is not None:
            self.fairness_train = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            self.fairness_train_t = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            self.fairness_val = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            self.fairness_val_t = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            self.fairness_test = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
            self.fairness_test_t = getattr(metrics, opts.fairness_type)(**opts.fairness_options)
        else:
            self.fair_met = None
            self.fair_met_val = None

        ################################################# Data Saving ########################################
        self.data = {}
        self.data['train'] = {'x':[], 'y':[], 's':[]}
        self.data['val'] = {'x':[], 'y':[], 's':[]}
        self.data['test'] = {'x':[], 'y':[], 's':[]}

    def training_step(self, batch, batch_idx):

        adv_opt, tgt_opt = self.optimizers()

        x, y, s, label = batch
        n = x.shape[0]
        z, out = self.model(x)
        # z, out = self.model(y)
        out_adv = self.model_adv(z)
        # import pdb; pdb.set_trace()

        adv_loss = self.criterion['adv_loss'](out_adv, s)

        # out_adv.detach()
        # adv_loss = self.criterion['adv_loss'](out_adv, s)
        tgt_loss = self.criterion['tgt_loss'](out, label.long())
        # adv_loss.detach()
        loss = (1 - self.opts.tau) * tgt_loss - self.opts.tau * adv_loss
        # loss_tgt = self.loss_tgt_trn(out, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_tgt_loss', tgt_loss, on_step=False, on_epoch=True, prog_bar=True)

        tgt_opt.zero_grad()
        self.manual_backward(loss)
        tgt_opt.step()

        if self.current_epoch > self.opts.num_init_epochs:
            for _ in range(self.opts.num_adv_train_iters):
                z, out = self.model(x)
                out_adv = self.model_adv(z)

                adv_loss = self.criterion['adv_loss'](out_adv, s)

                adv_opt.zero_grad()
                self.manual_backward(adv_loss)
                adv_opt.step()

            self.log('train_adv_loss', adv_loss, on_step=False, on_epoch=True, prog_bar=True)

        acc = self.acc_trn(out, label.long())
        self.log('train_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #############################################
        if (self.current_epoch + 1) % self.opts.check_val_every_n_epochs == 0:
            hsic_s = self.fairness_train(z, s, self.opts, self.rbfsigma)
            hsic_y = self.fairness_train_t(z, y, self.opts, self.rbfsigma)


        ####################### Data Saving #######################
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['train']['x'].append(x)
            self.data['train']['y'].append(y)
            self.data['train']['s'].append(s)

    def training_epoch_end(self, outputs):
        sch_adv, sch_tgt = self.lr_schedulers()
        sch_adv.step()
        sch_tgt.step()
        if (self.current_epoch + 1) % self.opts.check_val_every_n_epochs == 0:

            kcc_s, dep_s = self.fairness_train.compute()
            kcc_y, dep_y = self.fairness_train_t.compute()


            self.log('train_dep_s', dep_s, on_step=False, on_epoch=True)
            self.log('train_kcc_s', kcc_s, on_step=False, on_epoch=True)

            self.log('train_dep_y', dep_y, on_step=False, on_epoch=True)
            self.log('train_kcc_y', kcc_y, on_step=False, on_epoch=True)

            self.fairness_train.reset()
            self.fairness_train_t.reset()

        else:
            pass

    def validation_step(self, batch, batch_idx):
        x, y, s, label = batch
        n = x.size(0)

        z, out = self.model(x)
        # z, out = self.model(y)
        out_adv = self.model_adv(z)
        adv_loss = self.criterion['val_adv_loss'](out_adv, s)
        tgt_loss = self.criterion['val_tgt_loss'](out, label.long())
        loss = (1 - self.opts.tau) * tgt_loss - self.opts.tau * adv_loss
        # loss_tgt = self.loss_tgt_val(out, y)
        # import pdb; pdb.set_trace()
        # loss_adv = self.loss_adv_val(out_adv, s)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_tgt_loss', tgt_loss, on_step=False, on_epoch=True)
        self.log('val_adv_loss', adv_loss, on_step=False, on_epoch=True, prog_bar=True)


        acc = self.acc_val(out, label.long())
        self.log('val_tgt_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        #########################################################################################
        # np.savetxt(self.opts.out_dir + '/z_val.out', z.detach().cpu(), fmt='%10.5f')
        # np.savetxt(self.opts.out_dir + '/y_val.out', y.detach().cpu(), fmt='%10.5f')
        # np.savetxt(self.opts.out_dir + '/s_val.out', s.detach().cpu(), fmt='%10.5f')
        if self.fairness_val is not None:
            dep_s = self.fairness_val(z, s, self.opts, self.rbfsigma)
            dep_y = self.fairness_val_t(z, y, self.opts, self.rbfsigma, label=True)

        ################### Data Saving ###########################
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['val']['x'].append(x)
            self.data['val']['y'].append(y)
            self.data['val']['s'].append(s)

    def validation_epoch_end(self, outputs):

        kcc_s, dep_s = self.fairness_val.compute()
        kcc_y, dep_y = self.fairness_val_t.compute()

        self.log('val_dep_s', dep_s, on_step=False, on_epoch=True)
        self.log('val_kcc_s', kcc_s, on_step=False, on_epoch=True)

        self.log('val_dep_y', dep_y, on_step=False, on_epoch=True)
        self.log('val_kcc_y', kcc_y, on_step=False, on_epoch=True)

        self.fairness_val.reset()
        # self.fairness_val_1.reset()
        self.fairness_val_t.reset()

    def test_step(self, batch, batch_idx):
        x, y, s, label = batch
        n = x.shape[0]

        z, out = self.model(x)
        # z, out = self.model(y)
        out_adv = self.model_adv(z)
        adv_loss = self.criterion['test_adv_loss'](out_adv, s)
        tgt_loss = self.criterion['test_tgt_loss'](out, label.long())
        loss = (1 - self.opts.tau) * tgt_loss - self.opts.tau * adv_loss

        acc = self.acc_tst(out, label.long())
        #########################################################################################

        if self.data_flag is not None:
            if self.data_flag:
                self.log('test_test_loss', loss, on_step=False, on_epoch=True)
                self.log('test_test_tgt_loss', tgt_loss, on_step=False, on_epoch=True)
                self.log('test_test_adv_loss', adv_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_test_acc', acc, on_step=False, on_epoch=True)
                self.log('test_embedding_dim', self.r, on_step=False, on_epoch=True)

            else:
                self.log('test_train_loss', loss, on_step=False, on_epoch=True)
                self.log('test_train_tgt_loss', tgt_loss, on_step=False, on_epoch=True)
                self.log('test_train_adv_loss', adv_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log('test_train_acc', acc, on_step=False, on_epoch=True)

        if self.fairness_test is not None:
            dep_s = self.fairness_test(z, s, self.opts, self.rbfsigma)
            dep_y = self.fairness_test_t(z, y, self.opts, self.rbfsigma)

        #################### Data Saving ##############################
        self.data['test']['x'].append(x)
        self.data['test']['y'].append(y)
        self.data['test']['s'].append(s)


    def test_epoch_end(self, outputs):
        metrics_dict = {}
        kcc_s, dep_s = self.fairness_test.compute()
        kcc_y, dep_y = self.fairness_test_t.compute()

        if self.data_flag is not None:
            if self.data_flag:
                self.log('test_test_dep_s', dep_s, on_step=False, on_epoch=True)
                self.log('test_test_kcc_s', kcc_s, on_step=False, on_epoch=True)

                self.log('test_test_dep_y', dep_y, on_step=False, on_epoch=True)
                self.log('test_test_kcc_y', kcc_y, on_step=False, on_epoch=True)

                metrics_dict['test_test_dep_s'] = float(dep_s.cpu().numpy().astype(np.float128))
                metrics_dict['test_test_kcc_s'] = float(kcc_s.cpu().numpy().astype(np.float128))

                metrics_dict['test_test_dep_y'] = float(dep_y.cpu().numpy().astype(np.float128))
                metrics_dict['test_test_kcc_y'] = float(kcc_y.cpu().numpy().astype(np.float128))

                metrics_dict['test_test_tgt_loss'] = float(
                    self.criterion['test_tgt_loss'].compute().cpu().numpy().astype(np.float128))

                metrics_dict['test_acc'] = float( self.acc_tst.compute().cpu().numpy().astype(np.float128))

                metrics_dict['test_test_adv_loss'] = float(
                    self.criterion['test_adv_loss'].compute().cpu().numpy().astype(np.float128))

                metrics_dict['test_embedding_dim'] = int(self.r)

                ####################################### Data Saving ######################################
                for split_name, data in self.data.items():
                    # import pdb; pdb.set_trace()
                    x = torch.cat(data['x'], dim=0)
                    z, _ = self.model(x)

                    y = torch.cat(data['y'], dim=0)
                    s = torch.cat(data['s'], dim=0)

                    np.savetxt(self.opts.out_dir + f'/z_{split_name}.out', z.detach().cpu(), fmt='%10.5f')
                    np.savetxt(self.opts.out_dir + f'/y_{split_name}.out', y.detach().cpu(), fmt='%10.5f')
                    np.savetxt(self.opts.out_dir + f'/s_{split_name}.out', s.detach().cpu(), fmt='%10.5f')

                ###############################################################################

            else:
                self.log('test_train_dep_s', dep_s, on_step=False, on_epoch=True)
                self.log('test_train_kcc_s', kcc_s, on_step=False, on_epoch=True)

                self.log('test_train_dep_y', dep_y, on_step=False, on_epoch=True)
                self.log('test_train_kcc_y', kcc_y, on_step=False, on_epoch=True)

                metrics_dict['test_train_dep_s'] = float(dep_s.cpu().numpy().astype(np.float128))
                metrics_dict['test_train_kcc_s'] = float(kcc_s.cpu().numpy().astype(np.float128))

                metrics_dict['test_train_dep_y'] = float(dep_y.cpu().numpy().astype(np.float128))
                metrics_dict['test_train_kcc_y'] = float(kcc_y.cpu().numpy().astype(np.float128))

                metrics_dict['test_train_tgt_loss'] = float(
                    self.criterion['test_tgt_loss'].compute().cpu().numpy().astype(np.float128))

                metrics_dict['test_train_acc'] = float( self.acc_tst.compute().cpu().numpy().astype(np.float128))

                metrics_dict['test_train_tgt_loss'] = float(
                    self.criterion['test_tgt_loss'].compute().cpu().numpy().astype(np.float128))

                metrics_dict['test_train_adv_loss'] = float(
                    self.criterion['test_adv_loss'].compute().cpu().numpy().astype(np.float128))


        # Saving Z

        self.fairness_test.reset()
        self.fairness_test_t.reset()
        # self.criterion['test_tgt_loss'].reset()
        self.to_txt(**metrics_dict)

    def to_txt(self, **kwargs):
        random_seed = self.opts.manual_seed
        tau = self.opts.tau
        txt = f'{{"random_seed": {random_seed}, "tau": {tau}'
        # txt = f'{{"tau": {tau}'

        for key, value in kwargs.items():
            txt += f', "{key}": {value}'

        txt += '}\n'

        # print(txt)
        file_dir = self.opts.result_path
        # file_name = self.opts.results_txt_file
        file_name = 'ARL.txt'

        try:
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        except:
            pass

        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'a') as f:
            f.write(txt)

    def configure_optimizers(self):
        optimizer_enc_tgt = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)
        optimizer_adv = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model_adv.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)
        if self.opts.scheduler_method is not None:
            scheduler_enc_tgt = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer_enc_tgt, **self.opts.scheduler_options
            )
        if self.opts.scheduler_method is not None:
            scheduler_adv = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer_adv, **self.opts.scheduler_options
            )
            return [optimizer_adv, optimizer_enc_tgt], [scheduler_adv, scheduler_enc_tgt]
        else:
            return [optimizer_adv, optimizer_enc_tgt]