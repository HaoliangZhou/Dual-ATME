import os
import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class MEGC2019(torch.utils.data.Dataset):
    """MEGC2019 dataset class with 3 categories"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label[idx]

    def __len__(self):
        return len(self.imgPath)


class MEGC2019_SI_MeRoI(torch.utils.data.Dataset):  # Flow +  Flow
    """MEGC2019_SI dataset class with 3 categories and other side information"""

    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.imgPath1 = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.imgPath1.append(texts[0].replace('flow', 'flow_puzzle'))
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]), 'r').convert('RGB')
        img1 = Image.open("".join(self.imgPath1[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)
        return {"data": img, "data1": img1, "class_label": self.label[idx], 'db_label': self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)


class MEGC2019_SI_MeRoI_single(torch.utils.data.Dataset):  # Flow
    def __init__(self, imgList, transform=None):
        self.imgPath = []
        self.label = []
        self.dbtype = []
        with open(imgList, 'r') as f:
            for textline in f:
                texts = textline.strip('\n').split(' ')
                self.imgPath.append(texts[0])
                self.label.append(int(texts[1]))
                self.dbtype.append(int(texts[2]))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open("".join(self.imgPath[idx]), 'r').convert('RGB')
        # plt.imshow(img)
        # plt.show()
        if self.transform is not None:
            img = self.transform(img)
        return {"data": img, "class_label": self.label[idx], 'db_label': self.dbtype[idx]}

    def __len__(self):
        return len(self.imgPath)


