# hsic.py

import torch
import time
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict

import hal.models as models
import hal.losses as losses
import hal.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np
from scipy.linalg import eigh
import os
import math

__all__ = ['ARLCelebA']


class ARLCelebA(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.save_hyperparameters()
        self.opts = opts
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)
        ####################################################################################################

        self.rbfsigma = getattr(models, opts.gaussian_sigma)()

        #################################################################################
        #                                  Models                                       #
        #################################################################################
        self.model = getattr(models, opts.model_type)(**opts.model_options)
        # import pdb; pdb.set_trace()
        self.model_adv = getattr(models, opts.model_adv_type)(**opts.model_adv_options)

        #################################################################################
        #                          Losses and Accuracies                                #
        #################################################################################
        # Losses and Accuracies
        self.criterion = {}
        self.criterion['train_tgt_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['train_adv_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_tgt_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['val_adv_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_tgt_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_adv_loss'] = getattr(losses, opts.loss_type)(**opts.loss_options)
        self.criterion['test_loss'] = getattr(losses, 'logLoss')()

        self.acc_tgt_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_adv_trn = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)

        self.acc_tgt_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_adv_val = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)

        self.acc_tgt_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)
        self.acc_adv_tst = getattr(metrics, opts.evaluation_type)(**opts.evaluation_options)




        #################################################################################
        #                                   DPV Metric                                  #
        #################################################################################
        # DPV Metric
        opts.fairness_options1['num_y_classes'] = int(opts.model_options['nclasses'])
        opts.fairness_options1['num_s_classes'] = int(opts.model_adv_options['nclasses'])
        opts.fairness_options1['num_sensitive_att'] = 1

        self.fairness1_val = getattr(metrics, opts.fairness_type1)(**opts.fairness_options1)
        self.fairness1_test = getattr(metrics, opts.fairness_type1)(**opts.fairness_options1)

        '''
        #################################################################################
        #                                   NNCC Metric                                 #
        ################################################################################# 
        # NNCC Metric
        self.nncc = {}
        self.nncc['z_train'] = metrics.NNCC_Metric(hdl=4, lin_x=False, lin_s=False, opts=self.opts) # Dep(z,s)
        self.nncc['y_train'] = metrics.NNCC_Metric(hdl=4, lin_x=False, lin_s=False, opts=self.opts) # Dep(out,s)

        self.nncc['z_val'] = metrics.NNCC_Metric(hdl=4, lin_x=False, lin_s=False, opts=self.opts) # Dep(z,s)
        self.nncc['y_val'] = metrics.NNCC_Metric(hdl=4, lin_x=False, lin_s=False, opts=self.opts) # Dep(out,s)

        self.nncc['z_test'] = metrics.NNCC_Metric(hdl=4, lin_x=False, lin_s=False, opts=self.opts) # Dep(z,s)
        self.nncc['y_test'] = metrics.NNCC_Metric(hdl=4, lin_x=False, lin_s=False, opts=self.opts) # Dep(out,s)
        '''

        self.data = {}
        self.data['train'] = {'x': [], 'y': [], 's': []}
        self.data['val'] = {'x': [], 'y': [], 's': []}
        self.data['test'] = {'x': [], 'y': [], 's': []}

        self.metrics_dict = {}

    def training_step(self, batch, batch_indx):
        # if batch_indx > 10: return
        x, y, s = batch

        z, out = self.model(x)
        out_adv = self.model_adv(z)  # .detach())

        # opt_adv, opt_enc, opt_nncc_x, opt_nncc_y = self.optimizers()
        opt_adv, opt_enc = self.optimizers()

        s_int, s_onehot = self.format_s_int_onehot(s)

        # import pdb; pdb.set_trace()

        #######################################################################################
        #                                  Update Encoder                                     #
        #######################################################################################
        adv_loss = self.criterion['train_adv_loss'](out_adv, s_int.long())  # .detach()
        tgt_loss = self.criterion['train_tgt_loss'](out, y)
        adv_acc = self.acc_adv_trn(out_adv, s_int.int())

        loss = (1 - self.opts.tau) * tgt_loss - self.opts.tau * adv_loss
        acc_tgt = self.acc_tgt_trn(out, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_tgt_loss', tgt_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_tgt_acc', acc_tgt, on_step=False, on_epoch=True, prog_bar=True)

        opt_enc.zero_grad()
        self.manual_backward(loss)  # , retain_graph=True)
        opt_enc.step()

        # if self.current_epoch > self.opts.num_init_epochs:
        # for _ in range(self.opts.num_adv_train_iters):
        #######################################################################################
        #                                Update Adversary                                     #
        #######################################################################################
        z, _ = self.model(x)
        out_adv = self.model_adv(z)  # .detach())
        adv_loss = self.criterion['train_adv_loss'](out_adv, s_int.long())
        # adv_acc = self.acc_adv_trn(out_adv, s_int.int())

        self.log('train_adv_loss', adv_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_adv_acc', adv_acc, on_step=False, on_epoch=True, prog_bar=True)

        opt_adv.zero_grad()
        self.manual_backward(adv_loss)  # , retain_graph=True)
        opt_adv.step()

        '''  
        if self.current_epoch == self.opts.nepochs - 1:
            out_onehot = self.format_out_onehot(out)
            #######################################################################################
            #                                     Update NNCC                                     #
            #######################################################################################
            self.nncc['z_train'](z, s_onehot)
            self.nncc['y_train'](out_onehot, s_onehot)
        '''
        # Appending Z
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['train']['x'].append(x)
            self.data['train']['y'].append(y)
            self.data['train']['s'].append(s)

    def training_epoch_end(self, outputs):

        ################################## Saving Z #########################################
        # if self.current_epoch == self.opts.nepochs - 1:
        #     z = torch.cat(self.z['train'], dim=0)
        #     np.savetxt(os.path.join(self.opts.out_dir, 'z_train.out'), z.cpu().detach().numpy(), fmt='%10.5f')

        sch_dec, sch_enc = self.lr_schedulers()
        sch_enc.step()
        # if self.current_epoch > self.opts.num_init_epochs:
        sch_dec.step()

        '''
        if self.current_epoch == self.opts.nepochs - 1:
            print('\n\nCompute Train\n\n')
            train_dep_z = self.nncc['z_train'].compute()
            train_dep_y = self.nncc['y_train'].compute()

            self.log('train_NNCC_dep_z', train_dep_z, on_step=False, on_epoch=True, prog_bar=True)
            self.metrics_dict['train_NNCC_dep_z'] = float(train_dep_z.cpu().detach().numpy().astype(np.float128))
            self.log('train_NNCC_dep_y', train_dep_y, on_step=False, on_epoch=True, prog_bar=True)
            self.metrics_dict['train_NNCC_dep_y'] = float(train_dep_y.cpu().detach().numpy().astype(np.float128))

            self.nncc['z_train'].reset()
            self.nncc['y_train'].reset()
        '''

        self.acc_tgt_trn.reset()
        # self.criterion['train_tgt_loss'].reset()

    def validation_step(self, batch, batch_indx):
        # if batch_indx > 10: return
        x, y, s = batch

        z, out = self.model(x)
        out_adv = self.model_adv(z)

        s_int, s_onehot = self.format_s_int_onehot(s)
        y_onehot = self.format_y_onehot(y)

        #######################################################################################
        #                                        ARL                                          #
        #######################################################################################

        adv_loss = self.criterion['val_adv_loss'](out_adv, s_int.long())
        tgt_loss = self.criterion['val_tgt_loss'](out, y)

        loss = (1 - self.opts.tau) * tgt_loss - self.opts.tau * adv_loss

        acc_tgt = self.acc_tgt_val(out, y)
        acc_adv = self.acc_adv_val(out_adv, s_int.int())

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_tgt_loss', tgt_loss, on_step=False, on_epoch=True)
        self.log('val_adv_loss', adv_loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc_tgt, on_step=False, on_epoch=True)
        self.log('val_acc_adv', acc_adv, on_step=False, on_epoch=True)


        #######################################################################################
        #                                        DPV                                          #
        #######################################################################################
        DPV = self.fairness1_val(out, s_int)

        '''
        #######################################################################################
        #                                       NNCC                                          #
        #######################################################################################    
        if self.current_epoch == self.opts.nepochs-1:
            out_onehot = self.format_out_onehot(out)

            self.nncc['z_val'](z, s_onehot)
            self.nncc['y_val'](out_onehot, s_onehot)
        '''
        # Appending Z
        if self.current_epoch == self.opts.nepochs - 1:
            self.data['val']['x'].append(x)
            self.data['val']['y'].append(y)
            self.data['val']['s'].append(s)

    def validation_epoch_end(self, outputs):

        ################################## Saving Z #########################################
        # if self.current_epoch == self.opts.nepochs - 1:
        #     z = torch.cat(self.z['val'], dim=0)
        #     np.savetxt(os.path.join(self.opts.out_dir, 'z_val.out'), z.cpu().numpy(), fmt='%10.5f')

        ############################## log DPV Metric ##################################
        out, DPV_var, DPV_max = self.fairness1_val.compute()

        self.log(f'val_DPV_var', math.sqrt(float(DPV_var.cpu().numpy().astype(np.float128))))
        # self.log(f'val_DPV_max', float(DPV_max.cpu().numpy().astype(np.float128)))

        self.fairness1_val.reset()



        '''
        if self.current_epoch == self.opts.nepochs-1:
            print('\n\nCompute Validation\n\n')

            val_dep_z = self.nncc['z_val'].compute()
            val_dep_y = self.nncc['y_val'].compute()

            self.log('val_NNCC_dep_z', val_dep_z, on_step=False, on_epoch=True, prog_bar=True)
            self.metrics_dict['val_NNCC_dep_z'] = float(val_dep_z.cpu().detach().numpy().astype(np.float128))
            self.log('val_NNCC_dep_y', val_dep_y, on_step=False, on_epoch=True, prog_bar=True)
            self.metrics_dict['val_NNCC_dep_y'] = float(val_dep_y.cpu().detach().numpy().astype(np.float128))

            self.nncc['z_val'].reset()
            self.nncc['y_val'].reset()
        '''

        self.acc_tgt_val.reset()
        # self.criterion['val_tgt_loss'].reset()

    def test_step(self, batch, batch_idx):
        x, y, s = batch

        z, out = self.model(x)
        out_adv = self.model_adv(z)

        s_int, s_onehot = self.format_s_int_onehot(s)
        y_onehot = self.format_y_onehot(y)

        adv_loss = self.criterion['test_adv_loss'](out_adv, s_int.long())
        tgt_loss = self.criterion['test_tgt_loss'](out, y)

        acc_adv = self.acc_adv_tst(out_adv, s_int.int())
        acc_tgt = self.acc_tgt_tst(out, y)

        loss = (1 - self.opts.tau) * tgt_loss - self.opts.tau * adv_loss

        # log total loss
        self.criterion['test_loss'](loss, len(x))

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_tgt_loss', tgt_loss, on_step=False, on_epoch=True)
        self.log('test_adv_loss', adv_loss, on_step=False, on_epoch=True)
        self.log('test_tgt_acc', acc_tgt, on_step=False, on_epoch=True)
        self.log('test_adv_acc', acc_adv, on_step=False, on_epoch=True, prog_bar=True)

        # Update DPV Metric
        DPV = self.fairness1_test(out, s_int)

        '''
        # Update NNCC Metric
        out_onehot = self.format_out_onehot(out)
        self.nncc['z_test'](z, s_onehot)
        self.nncc['y_test'](out_onehot, s_onehot)
        '''

        # Appending Z
        self.data['test']['x'].append(x)
        self.data['test']['y'].append(y)
        self.data['test']['s'].append(s)

    def test_epoch_end(self, outputs):
        # metrics_dict = {}
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

        self.log('test_DPV_var', math.sqrt(float(DPV_var.cpu().numpy().astype(np.float128))))
        metrics_dict['test_DPV_var'] = math.sqrt(float(DPV_var.cpu().numpy().astype(np.float128)))

        # self.log(f'test_DPV_max', float(DPV_max.cpu().numpy().astype(np.float128)))
        # metrics_dict['test_DPV_max'] = float(DPV_max.cpu().numpy().astype(np.float128))

        self.fairness1_test.reset()


        ########################### log loss and accuracy ######################################
        metrics_dict[f'test_loss'] = float(self.criterion['test_loss'].compute().cpu().numpy().astype(np.float128))
        self.criterion['test_loss'].reset()

        metrics_dict['test_tgt_acc'] = float(self.acc_tgt_tst.compute().cpu().numpy().astype(np.float128))
        self.acc_tgt_tst.reset()

        metrics_dict['test_tgt_loss'] = float(
            self.criterion['test_tgt_loss'].compute().cpu().numpy().astype(np.float128))
        # self.criterion['test_tgt_loss'].reset()

        metrics_dict['test_adv_loss'] = float(
            self.criterion['test_adv_loss'].compute().cpu().numpy().astype(np.float128))
        # self.criterion['test_adv_loss'].reset()

        metrics_dict['test_adv_acc'] = float(self.acc_adv_tst.compute().cpu().numpy().astype(np.float128))
        self.acc_adv_tst.reset()

        self.to_txt(**metrics_dict)

    def format_y_onehot(self, y):
        y_onehot = torch.zeros(y.size(0), self.opts.model_options['nclasses'], device=y.device).scatter_(1, y.unsqueeze(
            1).type(torch.int64), 1)
        return y_onehot

    def format_out_onehot(self, out):
        out_int = out.argmax(axis=1)
        out_onehot = torch.zeros(out_int.size(0), out.size(1), device=out_int.device).scatter_(1, out_int.unsqueeze(1),
                                                                                               1)
        return out_onehot

    def format_s_int_onehot(self, s0):
        s_binary = (s0 + 1) / 2  # -1,+1 -> 0,+1

        # -x- -x- -> int
        # s_int = s_binary[:,0]*2 + s_binary[:,1]
        n = s0.shape[1]
        s_int = 0
        for i in range(n):
            s_int += s_binary[:, i] * (2 ** (n - i - 1))

        # int -> one-hot
        s_onehot = torch.zeros(s_int.size(0), 2 ** n, device=s_int.device).scatter_(1, s_int.unsqueeze(1).long(), 1)

        return s_int.cuda(), s_onehot.cuda()

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
        optimizer_dec = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model_adv.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)

        optimizer_enc = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)

        '''    
        optimizer_nncc_x = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.nncc_x.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)

        optimizer_nncc_y = getattr(torch.optim, self.opts.optim_method)(
            filter(lambda p: p.requires_grad, self.nncc_y.parameters()),
            lr=self.opts.learning_rate, **self.opts.optim_options)
        '''

        if self.opts.scheduler_method is not None:
            scheduler_enc = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer_enc, **self.opts.scheduler_options
            )
            scheduler_dec = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer_dec, **self.opts.scheduler_options
            )
            '''
            scheduler_nncc_x = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer_nncc_x, **self.opts.scheduler_options
            )
            scheduler_nncc_y = getattr(torch.optim.lr_scheduler, self.opts.scheduler_method)(
                optimizer_nncc_y, **self.opts.scheduler_options
            )
            '''
            # return [optimizer_dec, optimizer_enc, optimizer_nncc_x, optimizer_nncc_y], [scheduler_dec, scheduler_enc, scheduler_nncc_x, scheduler_nncc_y]
            return [optimizer_dec, optimizer_enc], [scheduler_dec, scheduler_enc]
        else:
            # return [optimizer_dec, optimizer_enc, optimizer_nncc_x, optimizer_nncc_y]
            return [optimizer_dec, optimizer_enc]