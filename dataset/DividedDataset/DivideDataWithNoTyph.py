import os
import numpy as np
import random
from tqdm import tqdm

# 划分规则超参数
# 6步预测6步
# 滑窗步长为3h
back = 6
forward = 6
step = 3

numpy_dir = "/home/sunpeng/data/yiyi/precipitation/numpy"

# 在数据根目录下创建一个txt用于记录全部数据的路径
shuffle_sample_list_file = numpy_dir + "/ShuffleSampleList.txt"
no_typh_shuffle_sample_list_file = numpy_dir + "/NoTyphShuffleSampleList.txt"

train_dataset = numpy_dir + "/NoTyphTrainDataset.txt"
val_dataset = numpy_dir + "/NoTyphValDataset.txt"
test_dataset = numpy_dir + "/NoTyphTestDataset.txt"


def filter_no_typh_rain_data():
    with open(no_typh_shuffle_sample_list_file, "w") as no_typh_shuffle_sample_file:
        with open(shuffle_sample_list_file, "r") as shuffle_sample_file:
            data_list = shuffle_sample_file.readlines()
            for i in tqdm(range(len(data_list))):
                sample_path_list = data_list[i].strip().split(",")

                typh_label = 0
                for j in range(len(sample_path_list)):
                    item = np.load(os.path.join(numpy_dir, sample_path_list[j].replace("\\", "/")))
                    typh_label += item["typhoon"]

                if typh_label == 0:
                    no_typh_shuffle_sample_file.write(data_list[i])

        shuffle_sample_file.close()
    no_typh_shuffle_sample_file.close()


# 划分训练集、验证集、测试集
# 分别写入TrainDataset.txt TestDataset.txt ValDataset.txt
# sample一共5870 train:test:val = 10:2:1
# 训练集4515 测试集 903 验证集  452
def divide_no_typh_data():
    with open(no_typh_shuffle_sample_list_file, "r") as no_typh_shuffle_sample_file:
        data_list = no_typh_shuffle_sample_file.readlines()
        print(len(data_list))
        train_list = data_list[0:4515]
        test_list = data_list[4515:5418]
        val_list = data_list[5418:5870]
        with open(train_dataset, "w") as train_dataset_file:
            train_dataset_file.writelines(train_list)
        train_dataset_file.close()
        with open(test_dataset, "w") as test_dataset_file:
            test_dataset_file.writelines(test_list)
        test_dataset_file.close()
        with open(val_dataset, "w") as val_dataset_file:
            val_dataset_file.writelines(val_list)
        val_dataset_file.close()


if __name__ == '__main__':
    # filter_no_typh_rain_data()
    divide_no_typh_data()
