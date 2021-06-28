import os, cv2, pickle, random
from os import listdir
import os.path as osp
import numpy as np
from torch.utils.data import Dataset

class LensflareDataset(Dataset):
    def __init__(self, opt_datasets, transform=False):
        super().__init__()
        self.opt = opt_datasets
        self.batch_size = opt_datasets['batch_size']
        self.patch_size = opt_datasets['patch_size']
        self.split = opt_datasets['split']
        self.transform = transform
        self.direction = opt_datasets['direction']
        self.path2a = opt_datasets['train_input_img']
        self.path2b = opt_datasets['train_label_img']
        self.convert_img_to_pt(key='train_input_img')
        self.convert_img_to_pt(key='train_label_img')
        self.img_list_path2a = os.listdir(self.path2a)
        self.img_list_path2b = os.listdir(self.path2b)
        
        if len(self.img_list_path2a) != len(self.img_list_path2b):
            print('Dataset length is not equal!')
            raise NotImplementedError
    
    def convert_img_to_pt(self, key):
        if self.opt[key][-1] == '/':
            self.opt[key] = self.opt[key][:-1]

        img_list = os.listdir(self.opt[key])
        need_convert = False

        for i in range(len(img_list)):
            _, ext = osp.splitext(img_list[i])
            if ext != '.pt':
                need_convert = True
                break
        if need_convert == False:
            self.image_list = img_list
            return
        
        new_dir_path = self.opt[key] + '_pt'
        if osp.exists(new_dir_path) and len(os.listdir(new_dir_path))==len(img_list):
            self.opt[key] = new_dir_path
            return

        os.makedirs(new_dir_path, exist_ok=True)

        for i in range(len(img_list)):
            base, ext = osp.splitext(img_list[i])
            src_path = osp.join(self.opt[key], img_list[i])
            dst_path = osp.join(new_dir_path, base+'.pt')             

            with open(dst_path, 'wb') as _f:
                img = cv2.imread(src_path)
                pickle.dump(img, _f)

        self.opt[key] = new_dir_path

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = pickle.load(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_image_pair(self, idx):
        path2a_path = osp.join(self.opt['train_input_img'], self.img_list_path2a[idx]).replace('png', 'pt')
        path2b_path = osp.join(self.opt['train_label_img'], self.img_list_path2b[idx]).replace('png', 'pt')

        # load img
        a = self.read_img(path2a_path) # uint8
        b = self.read_img(path2b_path) # uint8

        if self.split == 'train':
            a_patch, b_patch = self.get_patch(a, b, self.patch_size)
            
            if self.transform:
                a = self.transform(a_patch)
                b = self.transform(b_patch)

        if self.direction == 'b2a':
            return b, a
        else:
            return a, b

    def get_patch(self, a, b, ps, scale):
        # TODO(6/29): 이미지 전체를 packing해 모든 이미지 패치 추출
        a_h, a_w, a_c = a.shape
        b_h, b_w, b_c = b.shape
        
        a_x = random.randint(0, a_w - ps)
        a_y = random.randint(0, a_h - ps)
        b_x = a_x * scale
        b_y = a_y * scale
        
        a_patch = a[a_y:a_y+ps, a_x:a_x+ps, :]
        b_patch = b[b_y:b_y+ps*scale, b_x:b_x+ps*scale, :]

        return a_patch, b_patch

    def __getitem__(self, idx):
        start = (idx*self.batch_size) 
        end = start + self.batch_size

        if self.split == 'train':
            path2a_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
            path2b_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype=np.float32)
            
            for i in range(start, end):
                path2a, path2b = self.get_image_pair(i%len(self.img_list_path2a))
        else:
            path2a, path2b = self.get_image_pair(idx)
            path2a_batch, path2b_batch = np.expand_dims(path2a, 0), np.expand_dims(path2b, 0)

        return (path2a_batch).astype(np.float32), (path2b_batch).astype(np.float32)
        
    def __len__(self):
        return len([x for x in listdir(self.path2a)])