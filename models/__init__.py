from .pix2pix_unet import *

def create_model(args_network):
    device = torch.device('cuda:{}'.format(args_network['gpu_ids']) if torch.cuda.is_available() else 'cpu')
    which_model = args_network['which_model']
    
    if which_model == 'pix2pix_v1':
        model_gen = GeneratorUNet().to(device)
        model_dis = Discriminator().to(device)

        return model_gen, model_dis, device
    else:
        raise NotImplementedError('unrecognized model: {}'.format(which_model))
