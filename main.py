import matplotlib.pyplot as plt
import torch, argparse, time
from os import listdir
from os.path import join
from PIL import Image

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from configs import parse
from dataset import LensflareDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                                        [train_std[0], train_std[1], train_std[-1]]),
                    transforms.RandomHorizontalFlip(p=1), 
                    transforms.RandomVerticalFlip(p=1)
                    ])

    # Load dataset
    train_data = LensflareDataset(opt_datasets=args['datasets']['train'], transform=transform_train)
    lg.info('Create train dataset successfully!')
    lg.info('Training: [{}] iterations for each epoch'.format(len(train_data)))

    # 샘플 이미지 확인
    for i in range(1000):
        a,b = train_data[i]
        print(a.shape)
        time.sleep(1.5)