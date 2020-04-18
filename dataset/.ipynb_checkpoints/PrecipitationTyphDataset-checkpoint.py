import torch.utils.data
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random
import numpy as np

class PrecipitationTyphAreaDataset(torch.utils.data.Dataset):

    def __init__(self, root, file_name, if_train=False):
        super(PrecipitationTyphAreaDataset, self).__init__()
        self.root = root
        self.file_name = file_name
        self.if_train = if_train
        with open(os.path.join(root, file_name), 'r') as dataset_file:
            self.dataset_list = dataset_file.readlines()
        dataset_file.close()
        # print("init Dataset")

    def __getitem__(self, index):
        file_path_list = self.dataset_list[index].strip().split(",")
        precipitation = []
        typhoon = []
        mask = []
        for i in range(len(file_path_list)):
            item = np.load(os.path.join(self.root, file_path_list[i].replace("\\", "/")))
            # 降水台风3通道
            item_channel = []
            item_channel.append(item["precipitation"].squeeze().astype(np.float32))
            item_channel.append(item['typh_speed'].astype(np.float32))
            item_channel.append(item['typh_press'].astype(np.float32))
            item_channel = np.array(item_channel)  # [3,600,500]
            # print(item_channel.size())

            precipitation.append(item_channel)
            typhoon.append(item["typhoon"])
            mask.append(item["mask"])

        # list转换为numpy数组
        typhoon = np.array(typhoon)
        precipitation = np.array(precipitation)  # [12,3,600,500]
        mask = np.array(mask)

        # 数据增强
        if self.if_train:
            precipitation = precipitation.reshape(36, 600, 500).transpose(1, 2, 0)  # [600,500,36]
            mask = mask.squeeze().transpose(1, 2, 0)  # [600, 500, 12]
            precipitation, mask = self.my_transform(precipitation, mask)  # [600, 500, 36] [600, 500, 12]
            precipitation = precipitation.transpose(2, 0, 1)  # [36 ,600, 500]
            mask = mask.transpose(2, 0, 1)  # [12 ,600, 500]
            precipitation = precipitation.reshape(12, 3, 600, 500)  # [12, 3, 600, 500]
            mask = mask.unsqueeze(1)  # [12, 1, 600, 500]

        # print(precipitation)

        # numpy数组转换为torch
        precipitation, mask, typhoon = torch.from_numpy(precipitation), torch.from_numpy(mask), torch.from_numpy(
            typhoon)

        return precipitation[0:6], typhoon[0:6], precipitation[6:12, 0:1], typhoon[6:12], mask[6:12]


    def __len__(self):
        # print("getlen")
        return len(self.dataset_list)


def my_transform(self, image, mask, scale=(0.8, 1.2), crop_h=600, crop_w=500):
    # resize and random crop...
    temp_scale = scale[0] + (scale[1] - scale[0]) * random.random()
    temp_aspect_ratio = 1.0
    scale_factor_x = temp_scale * temp_aspect_ratio
    scale_factor_y = temp_scale / temp_aspect_ratio
    image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
    h, w = image.shape[0], image.shape[1]

    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                  cv2.BORDER_CONSTANT, value=0)
    h, w = image.shape[0], image.shape[1]

    h_off = random.randint(0, h - crop_h)
    w_off = random.randint(0, w - crop_w)
    image = image[h_off:h_off + crop_h, w_off:w_off + crop_w]
    mask = mask[h_off:h_off + crop_h, w_off:w_off + crop_w]

    # Flip
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask
