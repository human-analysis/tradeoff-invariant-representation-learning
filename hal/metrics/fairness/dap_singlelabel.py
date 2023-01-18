# demographic_parity.py
import torch
import torchmetrics.metric as tm
from typing import Any, Callable, Optional
import torch

__all__ = ['DP_SingleLabel']


class DP_SingleLabel(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 num_y_classes: int = 10,
                 num_s_classes: int = 4,
                 num_sensitive_att: int = 1
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("total", default=torch.zeros((num_sensitive_att, num_y_classes, num_s_classes)),
                       dist_reduce_fx=None)
        self.add_state("n_prob", default=torch.zeros((num_sensitive_att, num_y_classes, num_s_classes)),
                       dist_reduce_fx=None)

        self.num_s_classes = num_s_classes
        self.num_y_classes = num_y_classes
        self.num_sensitive_att = num_sensitive_att

    def update(self, yhat, list_s):
        '''
        yhat :: (batch_size, num_y_classes) : Prediction of the model
        list_s :: (batch_size, num_sensitive_att) : Sensitive Attributes

        Example:
            yhat = [
            [0.1, 0.3, 0.6],
            ...
            [0.7, 0.12, 0.18]
            ]

            list_s = [
            [0,2],
            ...
            [5,1]
            ]
        '''

        list_s = list_s.squeeze()
        yhat = yhat.squeeze()

        # assert list_s.shape[-1] == self.num_sensitive_att, f'Mismatch in list_s size and num_sensitive_att:: {list_s.shape[-1]} =/= {self.num_sensitive_att}'

        if len(yhat.shape) > 1:
            assert yhat.shape[
                       -1] == self.num_y_classes, f'Mismatch in yhat size and num_y_classes:: {yhat.shape[-1]} =/= {self.num_y_classes}'
            pred = yhat.data.max(1)[1]  # Extracts predicted class numbers
        else:
            assert self.num_y_classes == 1, f'Mismatch in yhat size and num_y_classes:: {self.num_y_classes} =/= 1'
            pred = yhat > 0.5  # Extracts predicted class numbers

        total = torch.zeros((self.num_sensitive_att, self.num_y_classes, self.num_s_classes))
        n_prob = torch.zeros((self.num_sensitive_att, self.num_y_classes, self.num_s_classes))

        if len(list_s.shape) > 1:  # If we have more than one sensitive attribute
            # import pdb; pdb.set_trace()
            assert list_s.shape[
                       -1] == self.num_sensitive_att, f'Mismatch in list_s size and num_sensitive_att :: {list_s.shape[-1]} =/= {self.num_sensitive_att}'
            for i in range(self.num_sensitive_att):
                s = list_s[:, i]
                for y in range(self.num_y_classes):
                    for s_0 in range(self.num_s_classes):
                        pred_0 = pred[s == s_0]  # Samples with s = s_0
                        total[i, y, s_0] = len(pred_0)  # Number of samples with s = s_0 { P(s=s_0) * #Batch_size}

                        n_prob[i, y, s_0] = len(pred_0[
                                                    pred_0 == y])  # Number of samples with yhat = y & s = s_0 { P(yhat=y, s=s_0) * #Batch_size}



        else:  # If we have only one sensitive attribute
            s = list_s[:]
            for y in range(self.num_y_classes):
                for s_0 in range(self.num_s_classes):
                    # import pdb; pdb.set_trace()
                    pred_0 = pred[s == s_0]  # Samples with s = s_0
                    total[0, y, s_0] = len(pred_0)  # Number of samples with s = s_0 { P(s=s_0) * #Batch_size}

                    n_prob[0, y, s_0] = len(pred_0[
                                                pred_0 == y])  # Number of samples with yhat = y & s = s_0 { P(yhat=y, s=s_0) * #Batch_size}

        total = total.to(device=self.total.device)
        n_prob = n_prob.to(device=self.n_prob.device)

        self.total += total
        self.n_prob += n_prob

        # import pdb; pdb.set_trace()

    def compute(self):

        prob = torch.zeros((self.num_sensitive_att, self.num_y_classes, self.num_s_classes), device=self.total.device)
        self.total = self.total.squeeze(0)
        self.n_prob = self.n_prob.squeeze(0)

        assert prob.shape == self.total.shape == self.n_prob.shape, f'Mismatch in prob, total, and n_prob sizes {prob.shape} =/= {self.total.shape} =/= {self.n_prob.shape}'

        # Compute P(y|s) = P(y,s) / P(s)

        tensor_size = self.total.shape
        for i in range(tensor_size[0]):
            for j in range(tensor_size[1]):
                for k in range(tensor_size[2]):

                    if self.total[i, j, k] > 0:
                        prob[i, j, k] = self.n_prob[i, j, k] / self.total[i, j, k]
                    else:
                        prob[i, j, k] = torch.Tensor([0]).to(device=self.total.device)

        # import pdb; pdb.set_trace()

        # Swap axes to make Y the first axis
        prob = torch.Tensor(prob.cpu().numpy().swapaxes(1, 0)).squeeze().to(device=self.total.device)

        # import pdb; pdb.set_trace()

        out = []
        sum_var = 0
        total_max = torch.Tensor([0]).to(device=self.total.device)

        # prob_y = [P(y=y_class|s=0), P(y=y_class|s=1), ..., P(y=y_class|s=num_s_classes-1)]
        for y_class, prob_y in enumerate(prob):
            var = prob_y.var()
            max = self._cross_diff(prob_y)
            out.append({'var': var, 'max': max})
            sum_var += var
            # import pdb; pdb.set_trace()
            if max > total_max:
                total_max = max

        total_var = sum_var / len(prob)

        # maximum_var = out[:, 0].max()
        # tot_max = out[:, 1].max()

        # tot_var = prob.var()

        # import pdb; pdb.set_trace()
        return out, total_var, total_max

    def _cross_diff(self, x):
        x = x.reshape(-1)

        sqx = torch.Tensor(len(x) * [x.cpu().numpy().tolist()]).to(
            device=self.total.device)  # Repeat x in rows in order to make a square tensor

        cross_diff_tensor = abs(sqx - sqx.transpose(1, 0))

        max = cross_diff_tensor.max()

        return max
