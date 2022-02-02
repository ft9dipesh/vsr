import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os

import model.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')

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
            self.netG.train()

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
        self.var_L = data['LQs'].to(self.device)
        self.real_H = data['GT'].to(self.device)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.var_L, self.real_H)

        b, c, h, w = self.real_H.shape
        l_pix = l_pix.sum() / int(b*c*h*w)

        l_pix.backward()
        self.optG.step()

        self.log_dict['l_pix'] = l_pix.item()

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.var_L.detach().float().cpu()
        else:
            out_dict['SR'] = self.var_L.detach().float().cpu()
            out_dict['INF'] = self.fake_H.detach().float().cpu()
            out_dict['HR'] = self.real_H.detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.var_L.detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
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
                self.fake_H = self.netG.super_resolution(self.var_L, continuous)
        self.netG.train()

    def sample(self, batch_size=1, continuous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continuous)
            else:
                self.SR = self.netG.sample(batch_size, continuous)
        self.netG.train()

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
