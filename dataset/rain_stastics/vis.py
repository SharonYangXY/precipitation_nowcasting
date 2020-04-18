import matplotlib
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

numpy_dir =  "/home/sunpeng/data/yiyi/precipitation/numpy"
typh_numpy_dir =  "/home/sunpeng/data/yiyi/precipitation/numpy/typh"
data_list_file = numpy_dir + "/DataList.txt"

with open(data_list_file, "r") as data_file:
    data_list = data_file.readlines()
    for i in range(len(data_list)):
        pre_path = os.path.join(numpy_dir, data_list[i].strip())
        typh_path = os.path.join(typh_numpy_dir, data_list[i].strip())
        typh_item = np.load(typh_path, allow_pickle=True)
        typh_point_list = typh_item["typh_point_list"]
        print(typh_point_list)
        if typh_point_list.any() == None:
            print(1)
            continue
        else:
            typh_label = len(typh_point_list)
            if typh_label > 0:
                pre_item = np.load(pre_path, allow_pickle=True)
                pre = pre_item["precipitation"]
                pre = pre.squeeze(0)

                print(data_list[i])

                a = pre
                a[pre >= 60] = 60
                # pre = a + pre
                pre = Image.fromarray(a.astype(np.uint8)*4)  # .convert('L')
                print(np.max(pre))
                # print(pre_item)
                # pre_numpy = np.array(pre)
                # cv2.imshow('pre', pre)
                plt.imshow(pre, plt.cm.jet)
                # plt.plot(pre)
                plt.show()
                #assert 0 == 1
