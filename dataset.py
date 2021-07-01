import os, cv2, pickle, random, time
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class LensflareDataset(Dataset):
    def __init__(self, opt_datasets, key, transform=False):
        super().__init__()
        self.key = key
        self.opt = opt_datasets
        self.batch_size = opt_datasets['batch_size']
        self.patch_size = opt_datasets['patch_size']
        self.stride = opt_datasets['stride']
        self.flip = opt_datasets['flip']
        self.rot = opt_datasets['rot']
        self.transform = transform
        self.split = opt_datasets['split']
        self.path2a = opt_datasets[self.key[0]]
        self.path2b = opt_datasets[self.key[-1]]
        self.convert_img_to_pt(key=self.key[0])
        self.convert_img_to_pt(key=self.key[-1])
        self.img_list_path2a = os.listdir(self.opt[self.key[0]])
        self.img_list_path2b = os.listdir(self.opt[self.key[-1]])
        
        if len(self.img_list_path2a) != len(self.img_list_path2b):
            print('Dataset length is not equal!')
            raise NotImplementedError
    
    def convert_img_to_pt(self, key):
        pt_path = self.opt[key].replace(key, key+'_pt')
        
        if osp.exists(pt_path) and len(os.listdir(pt_path)) != 0:
            self.opt[key] = pt_path
            return
        else:
            os.makedirs(pt_path, exist_ok=True)
            img_list = os.listdir(self.opt[key])

            for i in tqdm(range(len(img_list)), desc='Convert image into *.pt files({})...'.format(key)):
                base, ext = osp.splitext(img_list[i])
                src_path = osp.join(self.opt[key], img_list[i])
                img = cv2.imread(src_path)
                num = 0

                if self.split == 'train':
                    for top in range(0, img.shape[0], self.stride):
                        for left in range(0, img.shape[1], self.stride):
                            piece = np.zeros([self.patch_size, self.patch_size, 3], np.uint8)
                            temp = img[top:top+self.patch_size, left:left+self.patch_size, :]
                            piece[:temp.shape[0], :temp.shape[1], :] = temp
                            dst_path = osp.join(pt_path, base+'.pt')             
                            dst_path = f'{pt_path}{base}_{self.patch_size}_{num}.pt'

                            with open(dst_path, 'wb') as _f:
                                pickle.dump(piece, _f)
                                num+=1
                else:
                    dst_path = osp.join(pt_path, base+'.pt')
                    with open(dst_path, 'wb') as _f:
                            pickle.dump(img, _f)
                            num+=1

        self.opt[key] = pt_path

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = pickle.load(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_image_pair(self, idx, key):
        path2a_path = osp.join(self.opt[key[0]], self.img_list_path2a[idx]).replace('png', 'pt')
        path2b_path = osp.join(self.opt[key[-1]], self.img_list_path2b[idx]).replace('png', 'pt')

        # load img
        input = self.read_img(path2a_path) # uint8
        label = self.read_img(path2b_path) # uint8

        if self.split == 'train':
            input, label = self.augment(imageA=input, imageB=label, flip=self.flip, rot=self.rot)

        return input, label

    def augment(self, imageA, imageB, flip, rot):
        hflip = flip and random.random() < 0.5
        vflip = flip and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        if hflip:
            imageA = np.ascontiguousarray(imageA[:, ::-1, :])
            imageB = np.ascontiguousarray(imageB[:, ::-1, :])
        if vflip:
            imageA = np.ascontiguousarray(imageA[::-1, :, :])
            imageB = np.ascontiguousarray(imageB[::-1, :, :])
        if rot90:
            imageA = imageA.transpose(1, 0, 2)
            imageB = imageB.transpose(1, 0, 2)
    
        return imageA, imageB

    def __getitem__(self, idx):
        input, label = self.get_image_pair(idx, self.key)
        
        if self.transform:
            input = self.transform(input)
            label = self.transform(label)
        
        return input, label
        
    def __len__(self):
        return len(self.img_list_path2a)