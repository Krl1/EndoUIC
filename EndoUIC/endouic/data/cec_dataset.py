import glob
import random
import os
import pandas as pd
import cv2
import math
import numpy as np
import torch
import torch.utils.data as data
from basicsr.data import degradations as degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import normalize
from scripts.utils import pad_tensor, hiseq_color_cv2_img, generate_position_encoding

@DATASET_REGISTRY.register()
class CEC_Dataset(data.Dataset):

    def __init__(self, opt):
        super(CEC_Dataset, self).__init__()
        self.opt = opt

        self.gt_root = opt.get('gt_root')
        self.input_root = opt.get('input_root')
        self.csv_path = opt.get('csv_path')

        if self.csv_path and os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            self.gt_paths = df['clean'].tolist()
            self.input_paths = df['corrupted'].tolist()
        else:
            self.gt_paths = sorted(glob.glob(os.path.join(self.gt_root, '*.png')) + \
                            glob.glob(os.path.join(self.gt_root, '*.jpg')))
            self.input_paths = [os.path.join(self.input_root, os.path.split(v)[-1]) for v in self.gt_paths]

        self.mean = self.opt.get('mean', [0.5, 0.5, 0.5])
        self.std = self.opt.get('std', [0.5, 0.5, 0.5])

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        input_path = self.input_paths[index]

        gt_img = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255.
        input_img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB) / 255.

        if self.opt.get('bright_aug', False):
            bright_aug_range = self.opt.get('bright_aug_range', [0.5, 1.5])
            input_img = input_img * np.random.uniform(*bright_aug_range)
        
        if self.opt.get('concat_with_hiseq', False):
            hiseql = cv2.cvtColor(hiseq_color_cv2_img(cv2.imread(input_path)), cv2.COLOR_BGR2RGB) / 255.
            if self.opt.get('hiseq_random_cat', False) and np.random.uniform(0, 1) < self.opt.get('hiseq_random_cat_p', 0.5):
                input_img = np.concatenate([hiseql, input_img], axis=2)
            else:
                input_img = np.concatenate([input_img, hiseql], axis=2)
            
            if self.opt.get('random_drop', False):
                if np.random.uniform() <= self.opt.get('random_drop_p', 1.0):
                    random_drop_val = self.opt.get('random_drop_val', 0)
                    if np.random.uniform() < 0.5:
                        input_img[:, :, :3] = random_drop_val
                    else:
                        input_img[:, :, 3:] = random_drop_val
            if self.opt.get('random_drop_hiseq', False):
                if np.random.uniform() < 0.5:
                    input_img[:, :, 3:] = 0

        if self.opt.get('use_flip', False) and np.random.uniform() < 0.5:
            gt_img = cv2.flip(gt_img, 1)
            input_img = cv2.flip(input_img, 1)
        
        if self.opt.get('concat_with_position_encoding', False):
            H, W, _ = input_img.shape
            L = self.opt.get('position_encoding_L', 1)
            position_encoding = generate_position_encoding(H, W, L)
            input_img = np.concatenate([input_img, position_encoding], axis=2)
        
        if self.opt.get('resize', False):
            resize_size = self.opt['resize_size']
            gt_img = cv2.resize(gt_img, dsize=(resize_size[1], resize_size[0]))
            input_img = cv2.resize(input_img, dsize=(resize_size[1], resize_size[0]))

        if self.opt['input_mode'] == 'crop':
            # crop_size can be a single int or a list/tuple [h, w]
            crop_size = self.opt['crop_size']
            if isinstance(crop_size, int):
                crop_h, crop_w = crop_size, crop_size
            else:
                crop_h, crop_w = crop_size
            
            H, W, _ = input_img.shape
            assert input_img.shape[:2] == gt_img.shape[:2], f"{input_img.shape}, {gt_img.shape}, {gt_path}"
            
            h = np.random.randint(0, H - crop_h + 1)
            w = np.random.randint(0, W - crop_w + 1)
            gt_img = gt_img[h: h + crop_h, w: w + crop_w, :]
            input_img = input_img[h: h + crop_h, w: w + crop_w, :]

        if self.opt['input_mode'] == 'pad':
            divide = self.opt['divide']
            gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1))).float().unsqueeze(0)
            input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1))).float().unsqueeze(0)
            gt_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gt_img_pt, divide)
            input_img_pt, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_img_pt, divide)
            gt_img = gt_img_pt[0, ...].numpy().transpose((1, 2, 0))
            input_img = input_img_pt[0, ...].numpy().transpose((1, 2, 0))

        gt_img_pt = torch.from_numpy(gt_img.transpose((2, 0, 1))).float()
        input_img_pt = torch.from_numpy(input_img.transpose((2, 0, 1))).float()

        normalize(input_img_pt, [0.5] * input_img_pt.shape[0], [0.5] * input_img_pt.shape[0], inplace=True)
        normalize(gt_img_pt, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        
        return_dict = {"LR": input_img_pt, "HR": gt_img_pt, "lq_path": gt_path}
        if self.opt['input_mode'] == 'pad':
            return_dict.update({"pad_left": pad_left, "pad_right": pad_right, "pad_top": pad_top, "pad_bottom": pad_bottom})
        return return_dict

    def __len__(self):
        return len(self.gt_paths)
