import argparse, torch, os, warnings, cv2, time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from skimage.metrics import peak_signal_noise_ratio
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from configs import parse
from dataset import LensflareDataset
from models import create_model
from utils import ProgressBar, AverageMeter, fix_seed_everything, to_np, get_lr
import numpy as np

class Trainer(object):
    def __init__(self, args, writer, lg):
        self.args = args
        self.best_psnr = 0
        self.best_epoch = 0
        self.ckp_path = args['paths']['trained_models']

        if writer is not None:
            self.writer = writer # object for saving current status
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        if args['mode'] == 'train':
            lg.info('Start Training...!')
            transform = transforms.Compose([transforms.ToTensor()])

            # Load dataset
            train_data = LensflareDataset(opt_datasets=args['datasets']['train'], key=['train_input_img', 'train_label_img'], transform=transform)
            train_dataloader = DataLoader(train_data, batch_size=args['datasets']['train']['batch_size'], shuffle=True)
            lg.info('Create train dataset successfully!')
            lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))

            val_data = LensflareDataset(opt_datasets=args['datasets']['val'], key=['val_input_img', 'val_label_img'], transform=transform)
            val_dataloader = DataLoader(val_data, batch_size=args['datasets']['val']['batch_size'], shuffle=False)
            lg.info('Create validation dataset successfully!')
            lg.info('Validating: [{}] iterations for each epoch'.format(len(val_data)))

            # Define Generator and Discriminator
            model_gen, model_dis, device = create_model(args['networks'], lg)

            # Define Loss
            loss_func_gan = nn.BCELoss()
            loss_func_pix = nn.L1Loss()

            # Define Optimizers
            opt_dis = optim.Adam(model_dis.parameters(),lr=args['trainer']['lr'], 
                                                        betas=(args['trainer']['beta1'], 
                                                        args['trainer']['beta2']))
            opt_gen = optim.Adam(model_gen.parameters(),lr=args['trainer']['lr'],
                                                        betas=(args['trainer']['beta1'],
                                                        args['trainer']['beta2']))

            self.train_model(device=device, model_G=model_gen, model_D=model_dis, train_loader=train_dataloader, val_loader = val_dataloader,
                             adv_loss=loss_func_gan, l1_loss=loss_func_pix, opt_G=opt_gen, opt_D=opt_dis, lg=lg)

        elif args['mode'] == 'test':
            lg.info('Start Testing for Create Submit...!')
            
        else:
            print('Invalid mode')
            raise NotImplementedError

    def train_model(self, device, model_G, model_D, train_loader, val_loader, adv_loss, l1_loss, opt_G, opt_D, lg):
        if args['vis_type'] == 'wandb':
            wandb.watch(model_G)
            wandb.watch(model_D)
        
        model_G.train()
        model_D.train()

        # loss_func_pix weight
        lambda_pixel = 100

        # Define # of patches for patchGAN
        patch = (1,256//2**4,256//2**4)

        for epoch in range(self.args['trainer']['epochs']):
            pbar = ProgressBar(len(train_loader))
            epoch_g_loss = AverageMeter()
            epoch_d_loss = AverageMeter()
            # epoch_psnr = AverageMeter()

            for idx, (input, label) in enumerate(train_loader):
                ba_si = input.size(0)

                # real image
                real_input = input.to(device)
                real_label = label.to(device)

                # patch label
                real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
                fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

                # generator
                model_G.zero_grad()

                fake_generated = model_G(real_input) # generate lensflare removal image
                out_dis = model_D(fake_generated, real_label) # compare lensflare removal image VS. GT image

                gen_loss = adv_loss(out_dis, real_label)
                pixel_loss = l1_loss(fake_generated, real_label)

                g_loss = gen_loss + lambda_pixel * pixel_loss
                g_loss.backward()
                opt_G.step()

                # discriminator
                model_D.zero_grad()

                out_dis = model_D(real_label, real_input) # 진짜 이미지 식별
                real_loss = adv_loss(out_dis, real_label)
                
                out_dis = model_D(fake_generated.detach(), real_input) # 가짜 이미지 식별
                fake_loss = adv_loss(out_dis,fake_label)

                d_loss = (real_loss + fake_loss) / 2.
                d_loss.backward()
                opt_D.step()

                epoch_g_loss.update(g_loss.item())
                epoch_d_loss.update(d_loss.item())

                pbar.update('Training in Progress | Epoch: {} | Step: {} | G_Loss: {:.4f} | D_Loss: {:.4f}'.format(epoch, idx, epoch_g_loss.val, epoch_d_loss.val))

                if idx % args['log_interval_step'] == 0:
                    if args['vis_type'] == 'tensorboard':
                        self.writer.add_scalar('train/step_g_loss', epoch_g_loss.val, epoch*len(train_loader)+1)
                        self.writer.add_scalar('train/step_d_loss', epoch_d_loss.val, epoch*len(train_loader)+1)
                    elif args['vis_type'] == 'wandb':
                        wandb.log({'train/step_g_loss':epoch_g_loss.val, 'train/step_d_loss':epoch_d_loss.val})
                    else:
                        lg.info('Epoch/Step: {:4}/{:4} | G_Loss: {:.4f} | D_Loss: {:.4f} | opt_G_lr: {:.2e} | opt_D_lr: {:.2e}'.format(
                            epoch, idx, epoch_g_loss.val, epoch_d_loss.val, get_lr(opt_G), get_lr(opt_D)))

            epoch_psnr = self.on_epoch_end(model_G=model_G, model_D=model_D, device=device, val_dataloader=val_loader, epoch=epoch)

        if args['vis_type'] == 'tensorboard':
            self.writer.add_scalar('train/epoch_g_loss', epoch_g_loss.avg, epoch)
            self.writer.add_scalar('train/epoch_d_loss', epoch_d_loss.avg, epoch)
            self.writer.add_scalar('train/epoch_psnr', epoch_psnr.avg, epoch)
        elif args['vis_type'] == 'wandb':
            wandb.log({'train/epoch_g_loss':epoch_g_loss.avg, 'train/epoch_d_loss':epoch_d_loss.avg, 'train/epoch_avg_psnr':epoch_psnr.avg})
        else:
            lg.info('Epoch: {} | G_avg_Loss: {:.4f} | D_avg_Loss: {:.4f} | epoch_PSNR: {:.2f} | Best_PSNR: {:.2f} in Epoch [{}]'.format(
                            epoch, epoch_g_loss.avg, epoch_d_loss.avg, epoch_psnr.avg, self.best_psnr, self.best_epoch))
    
    def on_epoch_end(self, model_G, model_D, device, val_dataloader, epoch):
        epoch_psnr = AverageMeter()
        model_G.eval()
        model_D.eval()

        # Generate fake image
        with torch.no_grad():
            for a, b in val_dataloader:
                real_input = a
                real_label = b
                fake_imgs = model_G(a.to(device)).detach().cpu()

                if len(real_label) != len(fake_imgs):
                    print('I/O image length are not same!')
                    raise NotImplementedError
                break
            
            # Calculate batch PSNR score
            for idx in range(0, len(fake_imgs), 2):
                real_input_int8 = np.array(to_pil_image(0.5*real_input[idx]+0.5))
                real_label_int8 = np.array(to_pil_image(0.5*real_label[idx]+0.5))
                fake_generated_int8 = np.array(to_pil_image(0.5*fake_imgs[idx]+0.5))
                epoch_psnr.update(peak_signal_noise_ratio(real_label_int8[idx], fake_generated_int8[idx]))

                if idx % args['datasets']['val']['save_duration'] == 0:
                    if args['datasets']['val']['save_img'] == True:
                        result_img = np.hstack((real_input_int8, real_label_int8, fake_generated_int8))
                        cv2.imwrite(os.path.join(args['paths']['result_path'], '{}_epoch{}_{}_val.png'.format(args['modelname'], epoch, idx)), result_img)

        # save best status
        if epoch_psnr.val >= self.best_psnr:
            self.best_psnr = epoch_psnr.avg
            self.best_epoch = epoch
            model_save_base = self.ckp_path+'_'+str(epoch)+'epoch'
            torch.save(model_G.state_dict(), model_save_base+'_modelG.pt')
            torch.save(model_D.state_dict(), model_save_base+'_modelD.pt')
        
        return epoch_psnr

def main(args, lg, writer=None):
    Trainer(args, writer, lg)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='XLSR')
    parser.add_argument('--yaml_path', default='configs/pix2pix_lensflare.yaml')
    args = parser.parse_args()
    args, lg = parse(args)

    if args['vis_type'] == 'tensorboard':
        from tensorboardX import SummaryWriter
        tensorboard_path = args['paths']['visualizations'] + '/' + args['name']
        writer = SummaryWriter(tensorboard_path)
        main(args, lg, writer)
    elif args['vis_type'] == 'wandb':
        import wandb
        wandb.init(project="LensflareRemoval")
        wandb.config.update(args)
        main(args, lg, None)
    else:
        lg.info('No visualization tool initialized.')
        main(args, lg, None)

    fix_seed_everything(2021)
    