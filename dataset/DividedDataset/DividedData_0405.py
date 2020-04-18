import os
import numpy as np
import random

# 划分规则超参数
# 6步预测6步
# 滑窗步长为3h
back = 6
forward = 6
step = 4

numpy_dir = "/home/sunpeng/data/yiyi/precipitation/numpy"
typh_numpy_dir = "/home/sunpeng/data/yiyi/precipitation/numpy/typh"

# 在数据根目录下创建一个txt用于记录全部数据的路径
data_list_file = numpy_dir + "/DataList.txt"
sample_list_file = numpy_dir + "/SampleList_1h.txt"
shuffle_sample_list_file = numpy_dir + "/ShuffleSampleList_1h.txt"
typh_shuffle_sample_list_file = numpy_dir + "/TyphShuffleSampleList_1h.txt"

train_dataset = numpy_dir + "/TrainDataset_1h.txt"
val_dataset = numpy_dir + "/ValDataset_1h.txt"
test_dataset = numpy_dir + "/TestDataset_1h.txt"


# 按1h滑窗，记录所有sample ，一行为一个sample，包括back和forward
# 写入SampleList.txt文件
def write_sample_list():
    with open(data_list_file, "r") as data_file:
        data_list = data_file.readlines()
        with open(sample_list_file, "w") as sample_file:
            i = 0
            while i <= len(data_list) - 2 * (back + forward) + 1:
                for j in range(back + forward):
                    sample_file.write(data_list[i + 2 * j].strip())
                    if j != back + forward - 1:
                        sample_file.write(",")
                sample_file.write("\n")
                i += 1

            sample_file.close()
        data_file.close()


# 按step滑窗，记录相应sample，并乱序
# 写入ShuffleSampleList.txt文件
def shuffle_sample_list():
    with open(sample_list_file, "r") as sample_file:
        data_list = sample_file.readlines()
        sub_data_list = []
        for i in range(len(data_list)):
            if i % step == 0:
                sub_data_list.append(data_list[i])
        random.shuffle(sub_data_list)
        with open(shuffle_sample_list_file, "w") as shuffle_sample_file:
            shuffle_sample_file.writelines(sub_data_list)
        shuffle_sample_file.close()
    sample_file.close()


def filter_typh_rain_data():
    with open(typh_shuffle_sample_list_file, "w") as typh_shuffle_sample_file:
        with open(shuffle_sample_list_file, "r") as shuffle_sample_file:
            data_list = shuffle_sample_file.readlines()
            for i in range(len(data_list)):
                sample_path_list = data_list[i].strip().split(",")
                typh_label = 0
                for j in range(6):
                    item = np.load(os.path.join(typh_numpy_dir, sample_path_list[j].replace("\\", "/")), allow_pickle=True)
                    if item["typh_point_list"].any() == None:
                        continue
                    else:
                        typh_label += 1

                if typh_label >= 3:
                    print(typh_label)
                    typh_shuffle_sample_file.write(data_list[i])

        shuffle_sample_file.close()
    typh_shuffle_sample_file.close()


# 划分训练集、验证集、测试集
# 分别写入TrainDataset.txt TestDataset.txt ValDataset.txt
# sample一共3364 train:test:val = 10:2:1
# 训练集2588 测试集 517 验证集  259
def divide_typh_data():
    with open(typh_shuffle_sample_list_file, "r") as typh_shuffle_sample_file:
        data_list = typh_shuffle_sample_file.readlines()
        print(len(data_list))
        train_list = data_list[0:2588]
        test_list = data_list[2588:3105]
        val_list = data_list[3105:3364]
        with open(train_dataset, "w") as train_dataset_file:
            train_dataset_file.writelines(train_list)
        train_dataset_file.close()
        with open(test_dataset, "w") as test_dataset_file:
            test_dataset_file.writelines(test_list)
        test_dataset_file.close()
        with open(val_dataset, "w") as val_dataset_file:
            val_dataset_file.writelines(val_list)
        val_dataset_file.close()


if __name__ == "__main__":
    # write_sample_list()
    # shuffle_sample_list()
    # filter_typh_rain_data()
    divide_typh_data()
