import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import math


class large_dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(large_dataset, self).__init__()
        if train:
            root += '/ILSVRC2012_img_train'
        else:
            root += '/ILSVRC2012_img_val'
        self.transform = transform
        self.class_dir = os.listdir(root)
        self.data = []
        self.targets = []
        for target, dir_name in tqdm(enumerate(self.class_dir)):
            img_path_list = os.listdir(os.path.join(root, dir_name))
            for img_path in img_path_list:
                self.data.append(os.path.join(root, dir_name, img_path))
                self.targets.append(target)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img).convert("RGB").resize((256, 256))
        # img = np.array(img).reshape(1, 256, 256, 3)
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class pre_dataset(Dataset):
    def __init__(self, root, train=True, transform=None, save=False):
        super(pre_dataset, self).__init__()
        if train:
            root += '/ILSVRC2012_img_train'
        else:
            root += '/ILSVRC2012_img_val'
        self.transform = transform
        slice_num = 6

        if save:
            self.class_dir = os.listdir(root)
            self.data = []
            self.targets = []
            for target, dir_name in tqdm(enumerate(self.class_dir)):
                img_path_list = os.listdir(os.path.join(root, dir_name))
                for img_path in img_path_list:
                    img = Image.open(os.path.join(root, dir_name, img_path)).convert("RGB").resize((256, 256))
                    self.data.append(np.array(img).reshape(1, 256, 256, 3))
                    self.targets.append(target)
            self.data = np.vstack(self.data)
            self.targets = np.array(self.targets)
            pickle.dump(self.targets, open(root+'_targets.obj', 'wb'))
            if not train:
                pickle.dump(self.data, open(root+'_data.obj', 'wb'))
            else:
                train_len = len(self.data)
                slice_len = [train_len // slice_num + 1 if i < train_len % slice_num else train_len // slice_num for i in range(slice_num)]
                cur = 0
                for i in range(slice_num):
                    data = self.data[cur: cur+slice_len[i]]
                    cur += slice_len[i]
                    pickle.dump(data, open(root + '_data_%d.obj' % i, 'wb'))
        else:
            if train:
                data = []
                for i in range(slice_num):
                    data.append(pickle.load(open(root + '_data_%d.obj' % i, 'rb')))
                self.data = np.vstack(data)
            else:
                self.data = pickle.load(open(root+'_data.obj', 'rb'))
            self.targets = pickle.load(open(root+'_targets.obj', 'rb'))


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class ft_dataset(Dataset):
    def __init__(self, root, empty=False, transform=None, pre_cl_num=1000):
        super(ft_dataset, self).__init__()
        self.class_dir = os.listdir(root)
        self.data = []
        self.targets = []
        self.empty = empty
        self.transform = transform
        self.pre_cl_num = pre_cl_num
        if not self.empty:
            for target, dir_name in enumerate(self.class_dir):
                img_path_list = os.listdir(os.path.join(root, dir_name))
                for img_path in img_path_list:
                    img = Image.open(os.path.join(root, dir_name, img_path)).convert("RGB").resize((256, 256))
                    self.data.append(np.array(img).reshape(1, 256, 256, 3))
                    self.targets.append(target + self.pre_cl_num)
            self.data = np.vstack(self.data)
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
