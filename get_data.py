from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms


class Getdata(torch.utils.data.Dataset):
    def __init__(self, root):
        self.transform_norm = transforms.Compose([
            transforms.Resize((256, 200)),
            transforms.ToTensor()])
        self.transform_tensor = transforms.ToTensor()
        self.imageJ_path = osp.join(root, 'Watermarked_image', '%s.jpg')
        self.imageI_path = osp.join(root, 'Watermark_free_image', '%s.jpg')
        self.mask_path = osp.join(root, 'Mask', '%s.png')
        self.balance_path = osp.join(root, 'Loss_balance', '%s.png')
        self.alpha_path = osp.join(root, 'Alpha', '%s.png')
        self.W_path = osp.join(root, 'Watermark', '%s.png')
        self.root = root
        self.transform = transforms
        self.ids = list()
        for file in os.listdir(root + '/Watermarked_image'):
            # if(file[:-4]=='.jpg'):
            self.ids.append(file.strip('.jpg'))

    def __getitem__(self, index):
        imag_J, image_I, mask, balance, alpha, w = self.pull_item(index)
        return imag_J, image_I, mask, balance, alpha, w

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        img_J = Image.open(self.imageJ_path % img_id)
        img_I = Image.open(self.imageI_path % img_id)
        mask = Image.open(self.mask_path % img_id)
        balance = Image.open(self.balance_path % img_id)
        alpha = Image.open(self.alpha_path % img_id)
        w = Image.open(self.W_path % img_id)

        img_source = self.transform_norm(img_J)
        image_target = self.transform_norm(img_I)
        w = self.transform_norm(w)
        alpha = self.transform_norm(alpha)
        mask = self.transform_norm(mask)
        balance = self.transform_norm(balance)
        return img_source, image_target, mask, balance, alpha, w

if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    ds = Getdata('dataset/train')

    def get(i):
        past = time.time()
        img_J, img_I, mask, balance, alpha, w = ds[i]
        now = time.time()
        print(now - past)
        img_J, img_I, mask, balance, alpha, w = img_J.cpu(), img_I.cpu(), mask.cpu(), balance.cpu(), alpha.cpu(), w.cpu()
        img_J, img_I, mask, balance, alpha, w = img_J.permute(1, 2, 0), img_I.permute(1, 2, 0), mask.permute(1, 2, 0), \
            balance.permute(1, 2, 0), alpha.permute(1, 2, 0), w.permute(1, 2, 0)

        # print(img_J.shape, img_I.shape, mask.shape, balance.shape, alpha.shape, w.shape)
        # print(img_J)
        f, ax = plt.subplots(2, 3)
        ax[0][0].imshow(img_J)
        ax[0][1].imshow(img_I)
        ax[0][2].imshow(mask)
        ax[1][0].imshow(balance)
        ax[1][1].imshow(alpha)
        ax[1][2].imshow(w)

        plt.show()

    get(777)
    get(444)
    get(1616)
    get(555)