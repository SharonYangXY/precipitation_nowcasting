import numpy as np
import os
from tqdm import tqdm

numpy_dir = "/home/sunpeng/data/yiyi/precipitation/numpy"
numpy_save_dir = "/home/sunpeng/data/yiyi/precipitation/numpymask"


# 在数据根目录下创建一个txt用于记录全部数据的路径
data_list_file = numpy_dir + "/DataList.txt"


def mask2npz():
    # 默认缺省值-9999.9
    with open(data_list_file, "r") as data_file:
        data_list = data_file.readlines()
        for i in tqdm(range(len(data_list))):
            file_path = os.path.join(numpy_dir, data_list[i].replace("\\", "/").strip())
            item = np.load(file_path)
            precipitation = item["precipitation"][:, :, :500]
            typhoon = item["typhoon"]
            pre_array = precipitation
            mask = np.zeros_like(pre_array, dtype=np.int8)
            mask[pre_array >= 0] = 1
            np.savez(file_path, typhoon=typhoon, precipitation=precipitation, mask=mask)


mask2npz()
