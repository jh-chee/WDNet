from WDNet import generator
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
import os.path as osp
import os
import time
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str, default='models/WDNet_G_1.pth', help='Path to generator model')
parser.add_argument('--load_data', type=str, default='dataset/score_passport', help='Path to load images')
parser.add_argument('--result_dir', type=str, default='results', help='Path to save generated images')
args = parser.parse_args()

print(f'>> cuda: {torch.cuda.is_available()}')

G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(args.load_model, map_location='cuda:0'))
G.cuda()

imageJ_path = args.load_data
img_vision_path = args.result_dir
photo_files = sorted(os.listdir(imageJ_path))

print(f'>> models loaded from {args.load_model}')
print(f'>> photos loaded from {imageJ_path}')
print(f'>> results saved to {img_vision_path}')

# i = 0
# all_time = 0.0

for img_id in tqdm(photo_files):
    # i += 1
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_J = Image.open(osp.join(imageJ_path, img_id))
    img_source = transform_norm(img_J)
    img_source = torch.unsqueeze(img_source.cuda(), 0)
    # st = time.time()
    pred_target, mask, alpha, w, I_watermark = G(img_source)
    # all_time += time.time() - st
    # mean_time = all_time / i
    # print(f"img_id: {img_id}, mean time: {mean_time:.3f}")

    p0 = torch.squeeze(img_source)
    p1 = torch.squeeze(pred_target)
    p2 = mask
    p3 = torch.squeeze(w * mask)
    p2 = torch.squeeze(torch.cat([p2, p2, p2], 1))
    p0 = torch.cat([p0, p1], 1)
    p2 = torch.cat([p2, p3], 1)
    p0 = torch.cat([p0, p2], 2)
    p0 = transforms.ToPILImage()(p0.detach().cpu()).convert('RGB')
    # pred_target = transforms.ToPILImage()(p1.detach().cpu()).convert('RGB')

    """imshow for debugging"""
    # result = np.array(pred_target)
    # result = result[:, :, ::-1].copy()
    # vision = np.array(p0)
    # vision = vision[:, :, ::-1].copy()
    # # testsource = np.array(img_source.detach().cpu())
    # # testsource = testsource[:, :, ::-1].copy()
    # cv2.imshow('result', result)
    # cv2.imshow('vision', vision)
    # # cv2.imshow('source', testsource)
    # cv2.waitKey(0)
    # break

    # pred_target.save(osp.join(img_save_path, img_id))
    # if i <= 20:
    p0.save(osp.join(img_vision_path, img_id))
