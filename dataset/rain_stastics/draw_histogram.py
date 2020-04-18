import math
import os
from experiments.config import cfg
from matplotlib import pyplot as plt
import numpy as np


def drawHistogram(rain_dict, max_rain, min_rain):
    # TODO 绘制直方图

    max_x = math.ceil(max_rain)
    min_x = math.floor(min_rain)
    #
    # plt.hist(rain_dict, max_x, [min_x, max_x]);
    fig_path = os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, "降水量数据分布图.png")
    plt.savefig(fig_path)
    plt.show()


def calcuRainPercent(rain_dict):
    sum = 0.0
    rain_percent = {}
    for m in rain_dict.keys:
        sum += rain_dict[m]
    for n in rain_dict.keys:
        rain_percent[n] = rain_dict[n] / sum
    print(rain_dict)


if __name__ == '__main__':
    rain_data = np.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'typh-rain-statics.npz'), allow_pickle=True)
    rain_dict = rain_data["rain_statics"]
    max_rain = rain_data["max_rain"]
    min_rain = rain_data["min_rain"]
    calcuRainPercent(rain_dict)
    # drawHistogram(rain_dict, max_rain, min_rain)

# 降水统计结果
# {
#     '<0.5': 6552738145,
#     '0.5-1': 209871700,
#     '1-2': 188859833,
#     '2-5': 188275137,
#     '5-10': 83729199,
#     '10-20': 40316850,
#     '20-30': 10889080,
#     '30-50': 6646400,
#     '50-100': 2932481,
#     '>=100': 341175
# }


# 台风降水统计结果

# {
#     '<0.5': 2191076900,
#     '0.5-1': 76863719,
#     '1-2': 69387513,
#     '2-5': 71707746,
#     '5-10': 33891591,
#     '10-20': 17085806,
#     '20-30': 4794308,
#     '30-50': 3080301,
#     '50-100': 1510728,
#     '>=100': 201388
# }
