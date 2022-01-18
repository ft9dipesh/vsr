from collections import OrderedDict

import torch
import torch.nn as nn
import os

import model.networks as networks
from .base_model import BaseModel


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)

        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'],
            schedule_phase='train'
        )

        if self.opt['phase'] == 'train':
            selt.netG.train()

            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params,
                lr=opt['train']['optimizer']['lr']
            )
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)

        # Averaging if multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b*c*h*w)

        l_pix.backward()
        self.optG.step()

        self.log_dict['l_pix'] = l_pix.item()

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase !== schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt,
                    self.device,
                )
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def test(self, continuous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(self.data['SR'], continuous)
            else:
                self.SR = self.netG.super_resolution(self.data['SR'], continuous)
        self.netG.train()

    def sample(self, batch_size=1, continuous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continuous)
            else:
                self.SR = self.netG.sample(batch_size, continuous)
        self.netG.train()
