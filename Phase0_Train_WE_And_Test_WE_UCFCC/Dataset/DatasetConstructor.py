from PIL import Image
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import random
import time
import glob
import scipy.io as scio
import h5py
import math

class DatasetConstructor(data.Dataset):
    def __init__(self):
        return

    def get_path_tuple(self, i, dataset_name = "SHA"):
        if dataset_name == "SHA" or dataset_name == "SHB":
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
        elif dataset_name == "QNRF":
            img_name = "/img_" + ("%04d" % (i + 1)) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
        elif dataset_name == "UCF50": # just for testing
            test_list = []
            if self.scene_index == 1:
                test_list = [1, 2, 11, 19, 20, 21, 25, 33, 48, 50]
            elif self.scene_index == 2:
                test_list = [9, 10, 16, 18, 26, 27, 30, 40, 44, 47]
            elif self.scene_index == 3:
                test_list = [5, 13, 17, 22, 31, 38, 41, 42, 45, 49]
            elif self.scene_index == 4:
                test_list = [4, 6, 8, 14, 23, 28, 29, 34, 37, 39]
            elif self.scene_index == 5:
                test_list = [3, 7, 12, 15, 24, 32, 35, 36, 43, 46]
            else:
                raise ValueError('...')

            img_name = "/" + ("%d" % (test_list[i])) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(test_list[i]) + ".npy"
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        return img_name, gt_map_name

    def resize(self, img, dataset_name):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if dataset_name == "SHA" or dataset_name == "UCF50":
            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width
            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height
            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        elif dataset_name == "SHB":
            resize_height = height
            resize_width = width
        elif dataset_name == "QNRF":
            resize_height = 768
            resize_width = 1024
        else:
            raise NameError("No such dataset, only support SHA, SHB, QNRF")
        img = transforms.Resize([resize_height, resize_width])(img)
        return img


class TrainDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 train_num,
                 data_dir_path,
                 gt_dir_path,
                 mode='crop',
                 dataset_name="SHA",
                 device=None,
                 is_random_hsi=False,
                 is_flip=False,
                 fine_size = 400
                 ):
        super(TrainDatasetConstructor, self).__init__()
        self.train_num = train_num
        self.imgs = []
        self.fine_size = fine_size
        self.permulation = np.random.permutation(self.train_num)
        self.data_root, self.gt_root = data_dir_path, gt_dir_path
        self.mode = mode
        self.device = device
        self.is_random_hsi = is_random_hsi
        self.is_flip = is_flip
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))

        self.img_paths = glob.glob(self.data_root+"/*.jpg")

    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path = self.img_paths[self.permulation[index]]
            img = Image.open(img_path).convert("RGB")
            gt_map_path = os.path.join(self.gt_root, os.path.basename(img_path).replace(".jpg", ".npy"))
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path)))

            # Additional mask for worldexpo
            if img_path.find("104242-") != -1 or img_path.find("200247-") != -1:
                prefix_f = os.path.basename(img_path).split('-')[0]
            else:
                prefix_f = os.path.basename(img_path).split('_')[0]
            mask_path = os.path.join(self.data_root, prefix_f, "roi.mat")
            mask_path = mask_path.replace("train_frame", "train_label")
            mask_info = scio.loadmat(mask_path)
            pos_x = mask_info['maskVerticesXCoordinates']
            pos_y = mask_info['maskVerticesYCoordinates']
            pos = np.concatenate((pos_x, pos_y), 1).astype(np.int32)
            mask_tg = np.zeros((576, 720),dtype='uint8')
            cv2.fillPoly(mask_tg, [pos], 1) # fill 1 in the mask
            self.mask = mask_tg

            if self.is_random_hsi:
                img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(img)
            if self.is_flip:
                flip_random = random.random()
                if flip_random > 0.5:
                    img = F.hflip(img)
                    gt_map = F.hflip(gt_map)

            img, gt_map = transforms.ToTensor()(img), transforms.ToTensor()(gt_map)
            img_shape = img.shape  # C, H, W
            rh, rw = random.randint(0, img_shape[1] - self.fine_size), random.randint(0, img_shape[2] - self.fine_size)
            p_h, p_w = self.fine_size, self.fine_size
            img = img[:, rh:rh + p_h, rw:rw + p_w]
            gt_map = gt_map[:, rh:rh + p_h, rw:rw + p_w]
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = functional.conv2d(gt_map.view(1, 1, self.fine_size, self.fine_size), self.kernel, bias=None, stride=2, padding=0)
            # crop the mask
            mask_tg = torch.from_numpy(self.mask)
            mask_tg = mask_tg[rh:rh + p_h, rw:rw + p_w]
            mask_tg = mask_tg.view(1, 1, 400, 400).float()
            mask_tg_small = functional.interpolate(mask_tg, (200, 200), mode='nearest').view(1, 200, 200)
            mask_tg = mask_tg.view(1, 400, 400)
            return index, img.view(3, self.fine_size, self.fine_size), gt_map.view(1, 200, 200), mask_tg, mask_tg_small

    def __len__(self):
        return self.train_num

    def shuffle(self):
        self.permulation = np.random.permutation(self.train_num)
        return self

class EvalDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 scene_index, # just for UCF_CC_50 or worldexpo in this repo
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 mode="crop",
                 dataset_name="SHA",
                 device=None,
                 ):
        super(EvalDatasetConstructor, self).__init__()
        self.scene_index = scene_index # just for UCF_CC_50 or WorldExpo
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.mode = mode
        self.device = device
        self.dataset_name = dataset_name
        self.kernel = torch.FloatTensor(torch.ones(1, 1, 2, 2))

        # Additional mask for worldexpo (Need to copy 5 roi mats to gt_root)
        if self.dataset_name == 'WorldExpo':
            self.img_paths = glob.glob(self.data_root+"/*.jpg")
            mask_path = os.path.join(self.gt_root, "roi.mat")
            mask_info = scio.loadmat(mask_path)
            pos_x = mask_info['maskVerticesXCoordinates']
            pos_y = mask_info['maskVerticesYCoordinates']
            pos = np.concatenate((pos_x, pos_y), 1).astype(np.int32)
            mask_tg = np.zeros((576, 720),dtype='uint8')
            cv2.fillPoly(mask_tg, [pos], 1) # fill 1 in the mask
            self.mask = mask_tg
        elif self.dataset_name == 'UCF50':
            for i in range(self.validate_num):
                i_n, g_n = super(EvalDatasetConstructor, self).get_path_tuple(i, self.dataset_name)
                self.imgs.append([self.data_root + i_n, self.gt_root + g_n, i + 1])





    def __getitem__(self, index):
        if self.mode == 'crop':
            if self.dataset_name == 'WorldExpo':
                img_path = self.img_paths[index]
                gt_map_path = os.path.join(self.gt_root, os.path.basename(img_path).replace(".jpg", ".npy"))
            elif self.dataset_name == 'UCF50':
                img_path, gt_map_path, img_index = self.imgs[index]

            img = Image.open(img_path).convert("RGB")
            if self.dataset_name == 'WorldExpo':
                img_ori = transforms.Resize([576, 736])(img)
                img_ori_tensor = transforms.ToTensor()(img_ori)
            elif self.dataset_name == 'UCF50':
                img_ori = super(EvalDatasetConstructor, self).resize(img, self.dataset_name)
                img_ori_tensor = transforms.ToTensor()(img_ori)

            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path)))
            gt_map = transforms.ToTensor()(gt_map)

            if self.dataset_name == 'WorldExpo': # also need resize density map, as we do not resize density map as other datasets.
                gt_map = gt_map.unsqueeze(0)
                gt_map = functional.interpolate(gt_map, (576, 736), mode='bilinear').squeeze(0)

            img_shape, gt_shape = img_ori_tensor.shape, gt_map.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img_ori_tensor)

            # downsample the mask
            if self.dataset_name == 'WorldExpo':
                mask_tg = torch.from_numpy(self.mask).view(1, 1, 576, 720).float()
                mask_tg = functional.interpolate(mask_tg, (576, 736), mode='nearest')  # first, slightly largen the mask
                #                                        ---  736 is divided by 32 when testing, must be divided by 16 when training ---
                mask_tg_small = functional.interpolate(mask_tg, (576//2, 736//2), mode='nearest').view(1, 288, 368)
                mask_tg = mask_tg.view(1, 576, 736)
            elif self.dataset_name == 'UCF50':
                mask_tg = torch.ones(1, img_shape[1], img_shape[2])
                mask_tg_small = torch.ones(1, img_shape[1]//2, img_shape[2]//2)

            # For evaluation, because, the cropped mechanism, mask the input will degradation the performance
          #  img = img * mask_tg

            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs = []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])

            imgs = torch.stack(imgs)
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=2, padding=0)

           # here I also return img_path and original img
            img_index = index
            return img_path, img_ori_tensor, img_index, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2), mask_tg, mask_tg_small

    def __len__(self):
        return self.validate_num
