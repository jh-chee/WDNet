from PIL import Image
import numpy as np
import cv2
import os.path as osp
import os
import sys
import torch
from torchvision import datasets, transforms
import random


class Getdata(torch.utils.data.Dataset):
    def __init__(self, root):
        self.transform_img_to_tensor = transforms.Compose([
            # transforms.Resize((512, 400)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        self.transform_logo_to_tensor = transforms.Compose([
            transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        self.transform_tensor = transforms.ToTensor()
        self.root = root  # contains ground truth photos
        self.photo_path = osp.join(root, 'photos', '%s.jpg')
        self.watermark_path = osp.join(root, 'watermarks', '%s.png')
        self.len_watermarks = len(os.listdir(os.path.join(root, 'watermarks')))
        self.transform = transforms
        self.ids = []

        for file in os.listdir(os.path.join(root, 'photos')):
            self.ids.append(file.strip('.jpg'))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_J, img_I, mask, balance, alpha, w = self.generate_mask(img_id)

        # convert np to tensor
        # mask = self.transform_tensor(mask)
        # balance = self.transform_tensor(balance)
        # alpha = self.transform_tensor(alpha)

        return img_J, img_I, mask, balance, alpha, w

    def __len__(self):
        return len(self.ids)

    def generate_mask(self, img_id):
        '''
        Generates the following from photo and a random watermark
        Watermarked_image, Watermark_free_image, Mask, Loss_balance, Alpha, Watermark
        '''
        img = Image.open(self.photo_path % img_id)
        img = img.resize((200, 256))
        img_height, img_width = img.size  # 400, 512

        # generate logo
        logo_id = str(torch.randint(1, self.len_watermarks, (1,)).item())
        logo = Image.open(self.watermark_path % logo_id).convert('RGBA')
        # rotate_angle = random.randint(0, 360)
        rotate_angle = torch.randint(0, 360, (1,))
        logo_rotate = logo.rotate(rotate_angle, expand=True)
        logo_height = torch.randint(10, img_height, (1,))
        logo_width = torch.randint(10, img_width, (1,))
        logo_resize = logo_rotate.resize((logo_height, logo_width))
        img = self.transform_img_to_tensor(img)
        logo = self.transform_logo_to_tensor(logo_resize)

        img = img.cuda()
        logo = logo.cuda()

        alpha = torch.rand(1) * 0.4 + 0.3  # *0.4+0.3
        alpha = alpha.cuda()
        # print(f'{img_height - logo_height = }')

        start_height = torch.randint(0, img_height - logo_height.item(), (1,))
        start_width = torch.randint(0, img_width - logo_width.item(), (1,))
        img_I = img.clone()  # watermark free

        print(f'{alpha = }, {start_width = }, {logo_width = }, {start_height = }, {logo_height = }')

        # TODO: fix image plant issue for logo 177, 246
        img[:, start_width:start_width + logo_width, start_height:start_height + logo_height] *= \
            (1.0 - alpha * logo[3:4, :, :]) + logo[:3, :, :] * alpha * logo[3:4, :, :]

        img_J = img

        # print(((((1.0 - alpha) * logo[3:4, :, :]) * alpha * logo[3:4, :, :]) + logo[:3, :, :]).shape)
        # print(type(img), type(img_I))
        mask = solve_mask(img, img_I)
        mask_saved = torch.cat((mask[:, :, None], mask[:, :, None], mask[:, :, None]), 2) * 256.0

        balance = solve_balance(mask)
        # balance_saved = np.concatenate(
        #     (balance[:, :, np.newaxis], balance[:, :, np.newaxis], balance[:, :, np.newaxis]), 2
        # ) * 256.0
        balance_saved = torch.cat((balance[:, :, None], balance[:, :, None], balance[:, :, None]), 2) * 256.0

        W = torch.zeros_like(img)
        W[:, start_width:start_width + logo_width, start_height:start_height + logo_height] += logo[:3, :, :]

        alpha = alpha * mask
        # alpha = np.concatenate((alpha[:, :, np.newaxis], alpha[:, :, np.newaxis], alpha[:, :, np.newaxis]), 2) * 256.0
        alpha = torch.cat((alpha[:, :, None], alpha[:, :, None], alpha[:, :, None]), 2) * 256.0

        mask_saved = mask_saved.permute(2, 0, 1)
        balance_saved = balance_saved.permute(2, 0, 1)
        alpha = alpha.permute(2, 0, 1)
        # print(img_J.shape, img_I.shape, mask_saved.shape, balance_saved.shape, alpha.shape, W.shape)
        return img_J, img_I, mask_saved, balance_saved, alpha, W


def solve_mask(img, img_target):
    # img1 = np.asarray(img.permute(1, 2, 0).cpu())
    # img2 = np.asarray(img_target.permute(1, 2, 0).cpu())
    img3 = torch.abs(img - img_target)

    mask = img3.sum(0) > (15.0 / 255.0)
    mask = mask.long()

    # print(f'{img.shape = }')
    # print(f'{img3.shape = }')
    # print(f'{mask.shape = }')

    return mask


def solve_balance(mask):
    height, width = mask.shape
    k = int(mask.sum())

    mask2 = (1.0 - mask) * torch.rand((height, width)).cuda()
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

    ds = Getdata('dataset')

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

    get(1)
    get(2)
    get(50)
