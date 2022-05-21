# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import random
import torchvision.transforms as transforms
import os


identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, shot, transform, image_size=None, outlier_task=False, mixed_task=False, corrupted_task=False):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)
        
        n_shot = shot
        n_query = batch_size - n_shot
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        
        # mixed : support images and query images are from different classes
        self.mixed_task = mixed_task
        self.mix_dataloader_s, self.mix_dataloader_q = [], []
        for cl in self.cl_list:
            if corrupted_task:
                sub_dataset = CorruptedDataset(self.sub_meta[cl], cl, transform=transform, image_size=image_size)
            else:
                sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))
            if self.mixed_task:
                self.mix_dataloader_s.append(torch.utils.data.DataLoader(sub_dataset, batch_size=n_shot, shuffle=True, num_workers=0, pin_memory=False))
                self.mix_dataloader_q.append(torch.utils.data.DataLoader(RandomGaussian(image_size, n_query), batch_size=n_query, shuffle=True, num_workers=0, pin_memory=False))

        self.outlier_task = outlier_task
        if self.outlier_task:
            outlier_dataset = RandomGaussian(image_size, batch_size)
            self.sub_dataloader.append(torch.utils.data.DataLoader(outlier_dataset, **sub_data_loader_params))   

    def __getitem__(self, i):
        if self.mixed_task and i == len(self.cl_list):
            j, k = random.sample(range(len(self.cl_list)), 2)
            imgs_s, lbls_s = next(iter(self.mix_dataloader_s[j]))
            imgs_q, lbls_q = next(iter(self.mix_dataloader_q[k]))
            return torch.cat((imgs_s, imgs_q), 0), torch.cat((lbls_s, lbls_q), 0)

        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class CorruptedDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity, image_size=None, p=0.5):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.p = p
        self.n_corrupt = int(len(self.sub_meta) * self.p)

    def __getitem__(self, i):
        if i < len(self.sub_meta):
            image_path = os.path.join(self.sub_meta[i])
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
        else:
            img = torch.randn((3, self.image_size, self.image_size), dtype=torch.float)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta) + self.n_corrupt

class RandomGaussian:
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        return torch.randn((3, self.image_size, self.image_size), dtype=torch.float), 0  
        # label does not matter because it is reassigned to range(n_way)

class ZeroPixels:
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, i):
        return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float), 0

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, outlier_task=False, mixed_task=False, p=0.5):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.outlier_task = outlier_task
        self.mixed_task = mixed_task
        self.p = p

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        if self.outlier_task or self.mixed_task:
            for i in range(self.n_episodes):
                if random.random() < self.p:
                    yield torch.ones(self.n_way, dtype=torch.long) * self.n_classes
                else:
                    yield torch.randperm(self.n_classes)[:self.n_way]
        else:
            for i in range(self.n_episodes):
                yield torch.randperm(self.n_classes)[:self.n_way]
