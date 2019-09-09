import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

tsfm_aug = T.Compose([
                    # T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.111111)),
                    T.Resize((img_size, img_size)),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomAffine(degrees=10, shear=10),
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std),
                ])

tsfm_normal = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(rgb_mean, rgb_std),
        ])


class HCDDataset(Dataset):

    def __init__(self, path, indices=None, df=None, label_smooth=0, tta=False):
        """
        :param path: path where data files are stored
        :param split: split, one of 'TRAIN', 'VALID', or 'TEST'
        """
        self.path = path
        if df is not None:
            self.files = df['id'][indices].tolist()
            self.targets = df['label'][indices].values[:,None]
            if label_smooth>0:
                self.targets = np.fromiter(
                    [1-label_smooth if y==1 else label_smooth for y in self.targets], dtype=np.float32)[:,None]
        else:
            self.files = os.listdir(path+'test')
            self.targets = None
            self.tta = tta

    def __getitem__(self, index):
        if self.targets is not None:
            image = Image.open(self.path+'train/'+self.files[index]+'.tif', mode='r')
            image = image.convert('RGB')
            transform = tsfm_aug
            return transform(image), torch.FloatTensor(self.targets[index])
        else:
            image = Image.open(self.path+'test/'+self.files[index], mode='r')
            image = image.convert('RGB')
            if not self.tta:
                transform = tsfm_normal
            else:
                transform = tsfm_aug
            return transform(image)

    def __len__(self):
        return len(self.files)


