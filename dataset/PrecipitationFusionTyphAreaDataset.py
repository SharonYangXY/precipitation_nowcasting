import torch.utils.data
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random
import numpy as np


class PrecipitationFusionTyphAreaDataset(torch.utils.data.Dataset):

    def __init__(self, root, file_name, max_typhoon_number=2, if_train=False):
        super(PrecipitationFusionTyphAreaDataset, self).__init__()
        self.root = root
        self.file_name = file_name
        self.if_train = if_train
        self.max_typhoon_number = max_typhoon_number
        with open(os.path.join(root, file_name), 'r') as dataset_file:
            self.dataset_list = dataset_file.readlines()
        dataset_file.close()

    def __getitem__(self, index):
        file_path_list = self.dataset_list[index].strip().split(",")
        precipitation = []
        typhoon = []
        mask = []
        typh_point_list = []
        for i in range(len(file_path_list)):
            item = np.load(os.path.join(self.root, file_path_list[i].replace("\\", "/")), allow_pickle=True)
            item_typh = np.load(os.path.join(self.root + "/typh", file_path_list[i].replace("\\", "/")), allow_pickle=True)
            # precipitation typh 3 channel
            item_channel = []
            item_channel.append(item["precipitation"].squeeze().astype(np.float32))
            item_channel.append(item['typh_speed'].astype(np.float32))
            item_channel.append(item['typh_press'].astype(np.float32))
            item_channel = np.array(item_channel)  # [3,600,500]
            # print(item_channel.size())

            precipitation.append(item_channel)
            typhoon.append(item["typhoon"])
            mask.append(item["mask"])
            typh_point_list.append(item_typh["typh_point_list"])

        # list转换为numpy数组
        typhoon = np.array(typhoon)
        precipitation = np.array(precipitation)  # [12,3,600,500]
        mask = np.array(mask)
        typh_point_list = np.array(typh_point_list) #[12, 1, 7]
        
        gcn_masks = (self.get_gcn_mask_maps(typh_point_list))
        typh_gcn_masks = (self.get_typh_gcn_mask_maps(typh_point_list))
        
#         print("gcn_masks", gcn_masks.shape) # (6, 3, 600, 500)
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
        precipitation, mask, typhoon = torch.from_numpy(precipitation), torch.from_numpy(mask), torch.from_numpy(typhoon)
        gcn_masks = torch.from_numpy(gcn_masks)
        typh_gcn_masks = torch.from_numpy(typh_gcn_masks)

        return precipitation[0:6], typhoon[0:6], precipitation[6:12, 0:1], typhoon[6:12], mask[6:12], gcn_masks, typh_gcn_masks

    def __len__(self):
        # print("getlen")
        return len(self.dataset_list)
    
    def get_typh_gcn_mask_maps(self, typh_point_list, expand_pixels=3):
        #typh_point_list [12, 1, 7]
        input_typh_point_list = typh_point_list[0:6]
        all_input_masks = []
        for i in range(0, 6):
            temp_input_mask = []
            if input_typh_point_list[i].any() == None or len(input_typh_point_list[i]) == 0:
                temp_map = np.zeros((600, 500), dtype=np.float32)
                for t in range(0, self.max_typhoon_number):
                    temp_input_mask.append(temp_map)
            else:
                for j in range(0, input_typh_point_list[i].shape[0]):
                    if j > self.max_typhoon_number-1:
                        continue
                    x, y = input_typh_point_list[i][j][0], input_typh_point_list[i][j][1]
                    left_top_x = max(0, int(x-expand_pixels))
                    left_top_y = max(0, int(y-expand_pixels))
                    right_bot_x = min(600, int(x+expand_pixels))
                    right_bot_y = min(500, int(y+expand_pixels))
                    temp_map = np.zeros((600, 500), dtype=np.float32)
                    temp_map[left_top_x:right_bot_x,left_top_y:right_bot_y] = 1.0
                    temp_input_mask.append(temp_map)
                if input_typh_point_list[i].shape[0] < self.max_typhoon_number:
                    for t in range(0, self.max_typhoon_number-input_typh_point_list[i].shape[0]):
                        temp_map = np.zeros((600, 500), dtype=np.float32)
                        temp_input_mask.append(temp_map)
            temp_input_mask = np.array(temp_input_mask).astype(float)
#             print ("temp_input_mask", temp_input_mask.shape)
            all_input_masks.append(temp_input_mask)
        all_input_masks = np.array(all_input_masks)
        return all_input_masks
    
    def get_gcn_mask_maps(self, typh_point_list, expand_pixels=20):
        #typh_point_list [12, 1, 7]
        input_typh_point_list = typh_point_list[0:6]
#         print (input_typh_point_list.shape)
        all_input_masks = []
        for i in range(0, 6):
            temp_input_mask = []
            if input_typh_point_list[i].any() == None or len(input_typh_point_list[i]) == 0:
                temp_map = np.zeros((600, 500), dtype=np.float32)
                for t in range(0, self.max_typhoon_number):
                    temp_input_mask.append(temp_map)
            else:
                for j in range(0, input_typh_point_list[i].shape[0]):
                    if j > self.max_typhoon_number-1:
                        continue
                    x, y = input_typh_point_list[i][j][0], input_typh_point_list[i][j][1]
                    left_top_x = max(0, int(x-expand_pixels))
                    left_top_y = max(0, int(y-expand_pixels))
                    right_bot_x = min(600, int(x+expand_pixels))
                    right_bot_y = min(500, int(y+expand_pixels))
                    temp_map = np.zeros((600, 500), dtype=np.float32)
                    temp_map[left_top_x:right_bot_x,left_top_y:right_bot_y] = 1.0
                    temp_input_mask.append(temp_map)
                if input_typh_point_list[i].shape[0] < self.max_typhoon_number:
                    for t in range(0, self.max_typhoon_number-input_typh_point_list[i].shape[0]):
                        temp_map = np.zeros((600, 500), dtype=np.float32)
                        temp_input_mask.append(temp_map)
            temp_input_mask = np.array(temp_input_mask).astype(float)
#             print ("temp_input_mask", temp_input_mask.shape)
            all_input_masks.append(temp_input_mask)
        all_input_masks = np.array(all_input_masks)
        return all_input_masks


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


        
        
        
