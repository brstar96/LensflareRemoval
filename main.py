import argparse, torch, time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision.transforms as transforms
from configs import parse
from dataset import LensflareDataset
from models import create_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XLSR')
    parser.add_argument('--yaml_path', default='configs/pix2pix_lensflare.yaml')
    args = parser.parse_args()
    args, lg = parse(args)

    train_mean = args['datasets']['train']['mean']
    train_std = args['datasets']['train']['std']
    transform_train = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize([train_mean[0], train_mean[1], train_mean[-1]],
                                        [train_std[0], train_std[1], train_std[-1]]),])

    # Load dataset
    train_data = LensflareDataset(opt_datasets=args['datasets']['train'], transform=transform_train)
    train_dataloader = DataLoader(train_data, batch_size=args['datasets']['train']['batch_size'], shuffle=True)
    lg.info('Create train dataset successfully!')
    lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))

    # Define Generator and Discriminator
    model_gen, model_dis, device = create_model(args['networks'])

    # Define Loss
    loss_func_gan = nn.BCELoss()
    loss_func_pix = nn.L1Loss()

    # loss_func_pix 가중치
    lambda_pixel = 100

    # Define # of patches for patchGAN
    patch = (1,256//2**4,256//2**4)

    # Define Optimizers
    opt_dis = optim.Adam(model_dis.parameters(),lr=args['trainer']['lr'], 
                                                betas=(args['trainer']['beta1'], 
                                                args['trainer']['beta2']))
    opt_gen = optim.Adam(model_gen.parameters(),lr=args['trainer']['lr'],
                                                betas=(args['trainer']['beta1'],
                                                args['trainer']['beta2']))

    # Train Models
    model_gen.train()
    model_dis.train()

    batch_count = 0
    num_epochs = 100
    start_time = time.time()

    loss_hist = {'gen':[],
                'dis':[]}

    for epoch in range(num_epochs):
        for a, b in train_dataloader:
            ba_si = a.size(0)

            # real image
            real_a = a.to(device)
            real_b = b.to(device)

            # patch label
            real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

            # generator
            model_gen.zero_grad()

            fake_b = model_gen(real_a) # 가짜 이미지 생성
            out_dis = model_dis(fake_b, real_b) # 가짜 이미지 식별

            gen_loss = loss_func_gan(out_dis, real_label)
            pixel_loss = loss_func_pix(fake_b, real_b)

            g_loss = gen_loss + lambda_pixel * pixel_loss
            g_loss.backward()
            opt_gen.step()

            # discriminator
            model_dis.zero_grad()

            out_dis = model_dis(real_b, real_a) # 진짜 이미지 식별
            real_loss = loss_func_gan(out_dis,real_label)
            
            out_dis = model_dis(fake_b.detach(), real_a) # 가짜 이미지 식별
            fake_loss = loss_func_gan(out_dis,fake_label)

            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            opt_dis.step()

            loss_hist['gen'].append(g_loss.item())
            loss_hist['dis'].append(d_loss.item())

            batch_count += 1
            if batch_count % 100 == 0:
                print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))
