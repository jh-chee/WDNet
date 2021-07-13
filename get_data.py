from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms


class Getdata(torch.utils.data.Dataset):
    def __init__(self, photo_path, watermark_path, wallpaper_path):
        self.transform_img_to_tensor = transforms.Compose([
            transforms.Resize((256, 200)),
            transforms.RandomCrop(200),
            transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.5, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        self.transform_logo_to_tensor = transforms.Compose([
            transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

        self.photo_path = photo_path
        self.watermark_path = watermark_path
        self.wallpaper_path = wallpaper_path
        self.photo_files = []
        self.watermark_files = []
        self.wallpaper_files = []
        self.photo_files = sorted(os.listdir(self.photo_path))
        self.watermark_files = sorted(os.listdir(self.watermark_path))
        self.wallpaper_files = sorted(os.listdir(self.wallpaper_path))
        self.len_watermark = len(self.watermark_files)
        self.len_wallpaper = len(self.wallpaper_files)

    def __getitem__(self, index):
        img_id = self.photo_files[index]
        img_J, img_I, mask, balance, alpha, w = self.generate_mask(img_id)
        return img_J, img_I, mask, balance, alpha, w

    def __len__(self):
        return len(self.photo_files)

    def generate_mask(self, img_id):
        """
        Generates watermarked_image, Watermark_free_image, Mask, Loss_balance, Alpha, Watermark
        based on input photo, with a probability of 0.8 to load a random watermark
        """

        img = Image.open(osp.join(self.photo_path, img_id))
        img = self.transform_img_to_tensor(img)
        img_height, img_width = img.shape[2], img.shape[1]  # 200, 200
        img_I = img.clone()  # watermark free

        alpha = torch.rand(1) * 0.4 + 0.5
        W = torch.zeros_like(img)

        load_watermark = torch.rand(1) > 0.5
        if load_watermark:
            logo_id = self.watermark_files[torch.randint(1, self.len_watermark, (1,)).item()]
            logo = Image.open(osp.join(self.watermark_path, logo_id)).convert('RGBA')
            rotate_angle = torch.randint(0, 360, (1,))
            logo_height = torch.randint(10, img_height, (1,))
            logo_width = torch.randint(10, img_width, (1,))
            logo_rotate = logo.rotate(rotate_angle, expand=True)
            logo_resize = logo_rotate.resize((logo_height, logo_width))
            logo = self.transform_logo_to_tensor(logo_resize)

            start_height = torch.randint(0, img_height - logo_height.item(), (1,))
            start_width = torch.randint(0, img_width - logo_width.item(), (1,))

            # TODO: fix image plant issue for logo 177, 246
            img[:, start_width:start_width + logo_width, start_height:start_height + logo_height] *= \
                (1.0 - alpha * logo[3:4, :, :]) + logo[:3, :, :] * alpha * logo[3:4, :, :]

            W[:, start_width:start_width + logo_width, start_height:start_height + logo_height] += logo[:3, :, :]

        load_wallpaper = torch.rand(1) > 0.8
        if load_wallpaper:
            wallpaper_id = self.wallpaper_files[torch.randint(1, self.len_wallpaper, (1,)).item()]
            wallpaper = Image.open(osp.join(self.wallpaper_path, wallpaper_id)).convert('RGBA')

            wallpaper_resize = wallpaper.resize((img_height, img_width))
            wallpaper = self.transform_logo_to_tensor(wallpaper_resize)

            img *= (1.0 - alpha * wallpaper[3:4, :, :]) + wallpaper[:3, :, :] * alpha * wallpaper[3:4, :, :]

            W += wallpaper[:3, :, :]  # some values will be > 1
            W = W/torch.max(W)  # normalise the values to be between 0 and 1

        img_J = img  # with watermark

        mask = solve_mask(img_J, img_I)
        mask_saved = torch.cat((mask[:, :, None], mask[:, :, None], mask[:, :, None]), 2)

        balance = solve_balance(mask)
        balance_saved = torch.cat((balance[:, :, None], balance[:, :, None], balance[:, :, None]), 2)

        alpha = alpha * mask
        alpha = torch.cat((alpha[:, :, None], alpha[:, :, None], alpha[:, :, None]), 2)

        mask_saved = mask_saved.permute(2, 0, 1)
        balance_saved = balance_saved.permute(2, 0, 1)
        alpha = alpha.permute(2, 0, 1)
        return img_J, img_I, mask_saved, balance_saved, alpha, W


def solve_mask(img, img_target):
    img3 = torch.abs(img - img_target)
    mask = img3.sum(0) > (15.0 / 255.0)
    mask = mask.to(torch.float32)
    return mask


def solve_balance(mask):
    height, width = mask.shape
    k = int(mask.sum())

    mask2 = (1.0 - mask) * torch.rand((height, width))
    mask2 = mask2.flatten()
    pos = torch.argsort(mask2)
    balance = torch.zeros(height * width)
    balance[pos[:min(250 * 250, 4 * k)]] = 1
    balance = balance.reshape(height, width)
    return balance


# debugging
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    photo_path = 'dataset/photos'
    watermark_path = 'dataset/watermarks'
    wallpaper_path = 'dataset/wallpapers'
    ds = Getdata(photo_path, watermark_path, wallpaper_path)

    def get(i):
        past = time.time()
        img_J, img_I, mask, balance, alpha, w = ds[i]
        now = time.time()
        print(now - past)

        img_J, img_I, mask, balance, alpha, w = img_J.permute(1, 2, 0), img_I.permute(1, 2, 0), mask.permute(1, 2, 0), \
            balance.permute(1, 2, 0), alpha.permute(1, 2, 0), w.permute(1, 2, 0)
        print(torch.max(img_J), torch.max(img_I), torch.max(mask), torch.max(balance), torch.max(alpha), torch.max(w))
        print(img_J.shape, img_I.shape, mask.shape, balance.shape, alpha.shape, w.shape)
        print(img_J.dtype, img_I.dtype, mask.dtype, balance.dtype, alpha.dtype, w.dtype)

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
    get(1919)
    get(2934)
