import functools
import torch
import torch.nn as nn
from torch.nn import init
from . import unet


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.01):
    if init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(
            'Initialization method [{:s}] not implemented'.format(init_type)
        )


# Define Generator
def define_G(opt):
    model_opt = opt['model']

    from model.diffusion import GaussianDiffusion
    import model.EDVR.EDVR_arch as EDVR_arch

    model = EDVR_arch.EDVR(
        nf=model_opt['edvr']['nf'],
        nframes=model_opt['edvr']['nframes'],
        groups=model_opt['edvr']['groups'],
        front_RBs=model_opt['edvr']['front_RBs'],
        back_RBs=model_opt['edvr']['back_RBs'],
        center=model_opt['edvr']['center'],
        predeblur=model_opt['edvr']['predeblur'],
        HR_in=model_opt['edvr']['HR_in'],
        w_TSA=model_opt['edvr']['w_TSA']
    )
    # model = unet.UNet(
    #     in_channel=model_opt['unet']['in_channel'],
    #     out_channel=model_opt['unet']['out_channel'],
    #     norm_groups=model_opt['unet']['norm_groups'],
    #     inner_channel=model_opt['unet']['inner_channel'],
    #     channel_mults=model_opt['unet']['channel_multiplier'],
    #     attn_res=model_opt['unet']['attn_res'],
    #     res_blocks=model_opt['unet']['res_blocks'],
    #     dropout=model_opt['unet']['dropout'],
    #     image_size=model_opt['diffusion']['image_size'],
    # )

    netG = GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
    )
    if opt['phase'] == 'train':
        init_weights(netG, init_type='kaiming', scale=0.1)
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
