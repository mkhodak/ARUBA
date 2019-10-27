import math
import torch
from torch import nn
from torch import optim


class ARUBA(optim.Optimizer):

    def __init__(self, params, lr=1.0, collect=False):

        super(ARUBA, self).__init__(params, {})
        self.collect = collect

        if type(lr) == tuple:
            assert not collect, "cannot use ARUBA++ during meta-training"
            self.metastate, self.coef = lr
            for i, (group, metagroup) in enumerate(zip(self.param_groups, self.metastate)):
                for param, bsq, gsq in zip(group['params'], metagroup['bsq'], metagroup['gsq']):
                    state = self.state[param]
                    state['gsq'] = gsq.data.clone()
                    state['bsq'] = bsq
        else:
            self.coef = 0.0
            for i, group in enumerate(self.param_groups):
                etas = [lr] * len(group['params']) if type(lr) == float else lr[i]
                for param, eta in zip(group['params'], etas):
                    state = self.state[param]
                    state['eta'] = eta if type(lr) == float else eta.data
                    if collect:
                        state['phi'] = param.data.clone()
                        state['gsq'] = torch.zeros_like(param.data)


    def share_memory(self):

        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['eta'].share_memory()
                if collect:
                    state['phi'].share_memory()
                    state['gsq'].share_memory()
                elif self.coef:
                    state['bsq'].share_memory()
                    state['gsq'].share_memory()

    def step(self, closure=None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]
                if self.collect:
                    state['gsq'] += torch.pow(grad, 2)
                if self.coef:
                    eta = torch.sqrt(state['bsq'] / state['gsq'])
                    state['gsq'] += self.coef * torch.pow(grad, 2)
                    param.data -= eta * grad
                else:
                    param.data -= state['eta'] * grad

        return loss


class MetaOpt:

    def __init__(self, eps=1.0, zeta=1.0, p=1.0, lr=None):

        eps = eps if lr is None else lr * zeta
        self.eps = eps
        self.zeta = zeta
        self.eta = eps / zeta
        self.p = p
        self.state = None
        self.t = 1

    def update_state(self, optimizer):

        if optimizer.collect:
            if self.state is None:
                self.state = [{'bsq': [torch.full_like(param.data, self.eps ** 2) for param in group['params']],
                               'gsq': [torch.full_like(param.data, self.zeta ** 2) for param in group['params']]}
                               for group in optimizer.param_groups]
            self.bsq, self.gsq = 0.0, 0.0
            for state_group, group in zip(self.state, optimizer.param_groups):
                for b, g, param in zip(state_group['bsq'], state_group['gsq'], group['params']):
                    state = optimizer.state[param]
                    b += torch.pow(param.data - state['phi'], 2) / 2.0 + self.eps ** 2 / (self.t+1) ** self.p
                    g += state['gsq'] + self.zeta ** 2 / (self.t+1) ** self.p
                    self.bsq += float(torch.sum(b))
                    self.gsq += float(torch.sum(g))
        self.t += 1

    def update_eta(self, lr=None):
        
        if lr is None:
            if not self.state is None:
                self.eta = [[torch.sqrt(b / g) for b, g in zip(state_group['bsq'], state_group['gsq'])] for state_group in self.state]
        elif lr == 'iso':
            if not self.state is None:
                self.eta = math.sqrt(self.bsq / self.gsq)
        else:
            self.eta = lr

    def optimizer(self, params, coef=0.0, **kwargs):

        if coef and self.state is None:
            params = list(params)
            self.state = [{'bsq': [torch.full_like(param.data, self.eps ** 2) for param in params],
                           'gsq': [torch.full_like(param.data, self.zeta ** 2) for param in params]}]
        return ARUBA(params, lr=(self.state, coef) if coef else self.eta, **kwargs)
