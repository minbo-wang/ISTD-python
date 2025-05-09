import os.path as osp
from utils.images import load_image
import torch.utils.data as Data
import torchvision.transforms as transforms

import os
from PIL import Image


class MDFA(object):
    def __init__(self, base_dir='../data/MDFA/', mode='test'):
        assert mode in ['trainval', 'test']
        if mode == 'trainval':
            self.img_dir = osp.join(base_dir, 'training')
            self.mask_dir = osp.join(base_dir, 'training')
            self.length = 9978
        elif mode == 'test':
            self.img_dir = osp.join(base_dir, 'test_org')
            self.mask_dir = osp.join(base_dir, 'test_gt')
            self.length = 100
        else:
            raise NotImplementedError

        self.mode = mode

    def __getitem__(self, i):
        if self.mode == 'trainval':
            img_path = osp.join(self.img_dir, '%06d_1.png' % i)
            mask_path = osp.join(self.mask_dir, '%06d_2.png' % i)
        elif self.mode == 'test':
            img_path = osp.join(self.img_dir, '%05d.png' % i)
            mask_path = osp.join(self.mask_dir, '%05d.png' % i)
        else:
            raise NotImplementedError

        img = load_image(img_path)
        mask = load_image(mask_path)
        return img, mask

    def __len__(self):
        return self.length
    
class NUDTSIRST(object):
    def __init__(self, base_dir='../data/NUDT-SIRST/', mode='test'):
        self.img_dir = osp.join(base_dir, 'images')
        self.mask_dir = osp.join(base_dir, 'masks')
        self.length = 1235

        self.mode = mode

    def __getitem__(self, i):
        img_path = osp.join(self.img_dir, '%06d.png' % (i+1))
        mask_path = osp.join(self.mask_dir, '%06d.png' % (i+1))

        img = load_image(img_path)
        mask = load_image(mask_path)
        return img, mask

    def __len__(self):
        return self.length


class SIRST(object):
    def __init__(self, base_dir='../data/sirst/', mode='test'):
        if mode == 'trainval':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        return img, mask

    def __len__(self):
        return len(self.names)
    
class SirstAugDataset(Data.Dataset):
    def __init__(self, base_dir='../data/sirst_aug', mode='train', base_size=256):
        assert mode in ['train', 'test']

        if mode == 'train':
            self.data_dir = osp.join(base_dir, 'trainval')
        elif mode == 'test':
            self.data_dir = osp.join(base_dir, 'test')
        else:
            raise NotImplementedError

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, 'images')):
            if filename.endswith('png'):
                self.names.append(filename)

        self.img_transform = transforms.Compose([
            transforms.Resize((base_size, base_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((base_size, base_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])


    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, 'images', name)
        label_path = osp.join(self.data_dir, 'masks', name)

        img = load_image(img_path)
        mask = load_image(label_path)
        return img, mask

    def __len__(self):
        return len(self.names)
    
    @property
    def name(self):
        return 'sirstaug'