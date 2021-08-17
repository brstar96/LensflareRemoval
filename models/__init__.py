import os
from .pix2pix_unet import *

def create_model(args_network, lg):
    # Set GPU env
    GPU_NUM = args_network['gpu_ids']
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
    lg.info('Selected gpu: {} ({})'.format(GPU_NUM, torch.cuda.get_device_name(0)))
    lg.info('Allocated(init): {}GB, Cached: {}GB'.format(round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 
                                                   round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1)))

    which_model = args_network['which_model']
    
    if which_model == 'pix2pix_v1':
        model_gen = GeneratorUNet().to(device)
        model_dis = Discriminator().to(device)

        return model_gen, model_dis, device
    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))
