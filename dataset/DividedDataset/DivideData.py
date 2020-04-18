import os
import numpy as np
from netCDF4 import Dataset
import random

# 划分规则超参数
# 6步预测6步
# 滑窗步长为3h
back = 6
forward = 6
step = 3

dir = "D:\\Precipitation\\data\\1h_GPM"
numpy_dir = "D:\\Precipitation\\data\\1h_GPM\\numpy"

# 在数据根目录下创建一个txt用于记录全部数据的路径
data_list_file = numpy_dir + "\\DataList.txt"
sample_list_file = numpy_dir + "\\SampleList.txt"
shuffle_sample_list_file = numpy_dir + "\\ShuffleSampleList.txt"

train_dataset = numpy_dir + "\\TrainDataset.txt"
val_dataset = numpy_dir + "\\ValDataset.txt"
test_dataset = numpy_dir + "\\TestDataset.txt"


# 读取所有nc数据文件写入numpy文件
# 存入numpy文件夹下
def nc2npz():
    for year in os.listdir(dir):
        if year != "numpy":
            year_dir = os.path.join(dir, year)
            if year != "DataList.txt":
                for month in os.listdir(year_dir):
                    month_dir = os.path.join(year_dir, month)
                    for day in os.listdir(month_dir):
                        day_dir = os.path.join(month_dir, day)
                        for file in os.listdir(day_dir):
                            file_dir = os.path.join(day_dir, file)
                            numpy_file_dir = os.path.join(numpy_dir, year, month, day)
                            # 如果不存在则创建目录
                            #  创建目录操作函数
                            is_exists = os.path.exists(numpy_file_dir)
                            if not is_exists:
                                os.makedirs(numpy_file_dir)
                                print(numpy_file_dir + ' 创建成功')
                            nc_file = Dataset(file_dir)
                            typhoon = nc_file.variables['typhoon']
                            precipitation = nc_file.variables['precipitationCal'][:]
                            np.savez(numpy_file_dir + "\\" + file.split(".")[0] + ".npz", typhoon=typhoon,
                                     precipitation=precipitation)
                            nc_file.close()


# 记录所有数据相对路径
# 写入DataList.txt
def write_data_list():
    with open(data_list_file, "w") as f:
        # f.write(str_data)
        for year in os.listdir(numpy_dir):
            year_dir = os.path.join(numpy_dir, year)
            if year != "DataList.txt":
                for month in os.listdir(year_dir):
                    month_dir = os.path.join(year_dir, month)
                    for day in os.listdir(month_dir):
                        day_dir = os.path.join(month_dir, day)
                        for file in os.listdir(day_dir):
                            file_dir = os.path.join(year, month, day, file)
                            f.write(file_dir)
                            f.write('\n')

        f.close()


# 按1h滑窗，记录所有sample ，一行为一个sample，包括back和forward
# 写入SampleList.txt文件
def write_sample_list():
    with open(data_list_file, "r") as data_file:
        data_list = data_file.readlines()
        with open(sample_list_file, "w") as sample_file:
            for i in range(len(data_list) - back - forward + 1):
                for j in range(back + forward):
                    sample_file.write(data_list[i + j].strip())
                    if j != back + forward - 1:
                        sample_file.write(",")
                sample_file.write("\n")
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


# 划分训练集、验证集、测试集
# 分别写入TrainDataset.txt TestDataset.txt ValDataset.txt
# sample一共8756 train:test:val = 10:2:1
# 训练集6735 测试集 1347 验证集  674
def divide_data():
    with open(shuffle_sample_list_file, "r") as shuffle_sample_file:
        data_list = shuffle_sample_file.readlines()
        print(len(data_list))
        train_list = data_list[0:6735]
        test_list = data_list[6735:8082]
        val_list = data_list[8082:8756]
        with open(train_dataset,"w") as train_dataset_file:
            train_dataset_file.writelines(train_list)
        train_dataset_file.close()
        with open(test_dataset,"w") as test_dataset_file:
            test_dataset_file.writelines(test_list)
        test_dataset_file.close()
        with open(val_dataset,"w") as val_dataset_file:
            val_dataset_file.writelines(val_list)
        val_dataset_file.close()
