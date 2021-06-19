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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--load_model_dir', type=str, default='models/WDNet/WDNet_G_1.pth', help='Directory name for generator model')
parser.add_argument('--load_test_data', type=str, default='dataset/passport_score', help='Directory name to save generated images')
args = parser.parse_args()

print(torch.cuda.is_available())

G = generator(3, 3)
G.eval()
G.load_state_dict(torch.load(args.load_model_dir, map_location='cuda:0'))
G.cuda()

root = args.load_test_data
imageJ_path = osp.join(root, 'Watermarked_image', '%s.jpeg')
img_save_path = osp.join(root, 'results', 'result_img', '%s.jpeg')
img_vision_path = osp.join(root, 'results', 'result_vision', '%s.jpeg')

# if not os.path.exists(img_save_path):
#     os.makedirs(img_save_path)
#     print(f"{img_save_path} created")
# if not os.path.exists(img_vision_path):
#     os.makedirs(img_vision_path)
#     print(f"{img_vision_path} created")
#

ids = list()
for file in os.listdir(root + '/Watermarked_image'):
    # if(file[:-4]=='.jpg'):
    ids.append(file.strip('.jpeg'))

i = 0
all_time = 0.0

for img_id in ids:
    i += 1
    transform_norm = transforms.Compose([transforms.ToTensor()])
    img_J = Image.open(imageJ_path % img_id)
    img_source = transform_norm(img_J)
    img_source = torch.unsqueeze(img_source.cuda(), 0)
    st = time.time()
    pred_target, mask, alpha, w, I_watermark = G(img_source)
    all_time += time.time() - st
    mean_time = all_time / i
    print(f"img_id: {img_id}, mean time: {mean_time:.3f}")
    p0 = torch.squeeze(img_source)
    p1 = torch.squeeze(pred_target)
    p2 = mask
    p3 = torch.squeeze(w * mask)
    p2 = torch.squeeze(torch.cat([p2, p2, p2], 1))
    p0 = torch.cat([p0, p1], 1)
    p2 = torch.cat([p2, p3], 1)
    p0 = torch.cat([p0, p2], 2)
    p0 = transforms.ToPILImage()(p0.detach().cpu()).convert('RGB')
    pred_target = transforms.ToPILImage()(p1.detach().cpu()).convert('RGB')

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

    pred_target.save(img_save_path % img_id)
    # if i <= 20:
    p0.save(img_vision_path % img_id)
