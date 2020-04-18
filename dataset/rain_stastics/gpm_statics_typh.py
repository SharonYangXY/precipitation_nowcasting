from experiments.config import cfg
import os
import numpy as np
import threading
from tqdm import tqdm

numpy_dir = "/home/sunpeng/data/yiyi/precipitation/numpy"
data_list_dir = numpy_dir + "/DataList.txt"
typh_pre_list_Dir = numpy_dir + "/TyphPreDataList.txt"
typh_dataset_dir = numpy_dir + "/TyphShuffleSampleList.txt"


class MyThreadStatics(threading.Thread):
    def __init__(self, path_list):
        threading.Thread.__init__(self)  # 父类初始化
        self.path_list = path_list
        self.rain_dict = {
            '<0.5': 0,
            '0.5-1': 0,
            '1-2': 0,
            '2-5': 0,
            '5-10': 0,
            '10-20': 0,
            '20-30': 0,
            '30-50': 0,
            '50-100': 0,
            '>=100': 0
        }
        self.min_rain = 100.0
        self.max_rain = 0.0

    def run(self):
        for i in tqdm(range(len(self.path_list))):
            data_file_name = self.path_list[i].replace("\\", "/").strip()
            data = np.load(os.path.join(cfg.RAIN.ROOT, data_file_name))
            print("统计：", data_file_name)
            if data["typhoon"][0] != 0:
                rain_data = data["precipitation"][0][:, 0:500].flatten()
                for j in range(len(rain_data)):
                    # 判断最大最小值
                    if self.max_rain < rain_data[j]:
                        self.max_rain = rain_data[j]
                    if self.min_rain > rain_data[j]:
                        self.min_rain = rain_data[j]
                    # 统计字典区间
                    if rain_data[j] < 0.5:
                        self.rain_dict['<0.5'] += 1
                    elif rain_data[j] < 1:
                        self.rain_dict['0.5-1'] += 1
                    elif rain_data[j] < 2:
                        self.rain_dict['1-2'] += 1
                    elif rain_data[j] < 5:
                        self.rain_dict['2-5'] += 1
                    elif rain_data[j] < 10:
                        self.rain_dict['5-10'] += 1
                    elif rain_data[j] < 20:
                        self.rain_dict['10-20'] += 1
                    elif rain_data[j] < 30:
                        self.rain_dict['20-30'] += 1
                    elif rain_data[j] < 50:
                        self.rain_dict['30-50'] += 1
                    elif rain_data[j] < 100:
                        self.rain_dict['50-100'] += 1
                    else:
                        self.rain_dict['>=100'] += 1


def data_distribution():
    # [0.5, 1, 2, 5, 10, 20, 30, 50, 100, >100]
    # 左闭右开区间
    rain_dict = {
        '<0.5': 0,
        '0.5-1': 0,
        '1-2': 0,
        '2-5': 0,
        '5-10': 0,
        '10-20': 0,
        '20-30': 0,
        '30-50': 0,
        '50-100': 0,
        '>=100': 0
    }
    max_rain = 0.0
    min_rain = 100.0
    # rain_statics, max_rain, min_rain = gpm_data_statics(rain_dict)

    file_path = os.path.join(cfg.RAIN.ROOT, cfg.RAIN.DATA_LIST_file)
    with open(file_path, "r") as data_list_file:
        data_list = data_list_file.readlines()
        print(len(data_list))  # 26280个文件
        # 分72个线程处理
        # 每个进程处理365个文件
        thread_list = []
        for i in range(72):
            path_list = data_list[i * 365:(i + 1) * 365]
            static_thread = MyThreadStatics(path_list)
            static_thread.start()
            thread_list.append(static_thread)

        for thd in thread_list:
            thd.join()

        for thd in thread_list:
            if max_rain < thd.max_rain:
                max_rain = thd.max_rain
            if min_rain > thd.min_rain:
                min_rain = thd.min_rain
            rain_dict['<0.5'] += thd.rain_dict['<0.5']
            rain_dict['0.5-1'] += thd.rain_dict['0.5-1']
            rain_dict['1-2'] += thd.rain_dict['1-2']
            rain_dict['2-5'] += thd.rain_dict['2-5']
            rain_dict['5-10'] += thd.rain_dict['5-10']
            rain_dict['10-20'] += thd.rain_dict['10-20']
            rain_dict['20-30'] += thd.rain_dict['20-30']
            rain_dict['30-50'] += thd.rain_dict['30-50']
            rain_dict['50-100'] += thd.rain_dict['50-100']
            rain_dict['>=100'] += thd.rain_dict['>=100']

    # 保存统计结果
    rain_statics_path = cfg.GLOBAL.MODEL_SAVE_DIR + "/" + "typh-rain-statics.npz"

    np.savez(rain_statics_path, rain_statics=rain_dict,
             max_rain=max_rain, min_rain=min_rain)
    item = np.load(rain_statics_path, allow_pickle=True)
    print(item['rain_statics'])
    print(item['max_rain'])
    print(item['min_rain'])

    # 绘制直方图
    # max_x = math.ceil(max(x))
    # min_x = math.floor(min(x))

    # plt.hist(x, max_x, [min_x, max_x]);
    # fig_path = os.path.join(cfg.RAIN.ROOT, "降水量数据分布图.png")
    # plt.savefig(fig_path)
    # plt.show()


def write_typh_data_list():
    typh_data_list = []
    with open(typh_pre_list_Dir, "w") as typh_pre_list_file:
        with open(typh_dataset_dir, "r") as typh_dataset_file:
            typh_sample_list = typh_dataset_file.readlines()
            for i in tqdm(range(len(typh_sample_list))):
                item_list = typh_sample_list[i].strip().split(",")
                for j in range(len(item_list)):
                    if item_list[j] not in typh_data_list:
                        typh_data_list.append(item_list[j])
                        typh_pre_list_file.write(item_list[j])
                        typh_pre_list_file.write('\n')
                        print(item_list[j])
        typh_dataset_file.close()
        print(len(typh_data_list))


def TyphPreDataList():
    pixels = []
    with open(typh_pre_list_Dir, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            img_path = os.path.join(numpy_dir, lines[i].replace("\\", "/").strip())
            data = np.load(img_path)
            rain_data = data["precipitation"][0].flatten()
            pixels.extend([x / 250 for x in rain_data if x >= 0])

    means = np.mean(pixels)
    std = np.str(pixels)
    # # 保存统计结果
    # rain_statics_path = cfg.GLOBAL.MODEL_SAVE_DIR + "/" + "typh-rain-statics.npz"
    # np.savez(rain_statics_path, means=means, std=std, allow_pickle=True)
    # print(item['rain_statics'])
    # print(item['max_rain'])
    # print(item['min_rain'])


if __name__ == '__main__':
    # data_distribution()
    # write_typh_data_list()
    # TyphPreDataList()
    TyphPreDataList()

    # 最大值为240
    # rain_statics_path = cfg.GLOBAL.MODEL_SAVE_DIR + "/" + "typh-rain-statics.npz"
    # data = np.load(rain_statics_path)
    # print(data["max_rain"])
