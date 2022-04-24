from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 8
__C.GLOBAL.MODEL_SAVE_DIR = '/home/yxy/save_nowcasting'

__C.RAIN = edict()

# GPU
__C.RAIN.ROOT = "/home/yxy/data/GPM/numpy"
__C.RAIN.MultiChannelROOT = "/home/yxy/data/GPM/numpy"
__C.RAIN.TRAIN_file_NAME = "TrainDataset.txt"
__C.RAIN.VAL_file_NAME = "ValDataset.txt"
__C.RAIN.TEST_file_NAME = "TestDataset.txt"
__C.RAIN.DATA_LIST_file = "DataList.txt"

__C.RAIN.TYPH_TRAIN_file_NAME = "TyphTrainDataset.txt"
__C.RAIN.TYPH_VAL_file_NAME = "TyphValDataset.txt"
__C.RAIN.TYPH_TEST_file_NAME = "TyphTestDataset.txt"

__C.RAIN.NO_TYPH_TRAIN_file_NAME = "NoTyphTrainDataset.txt"
__C.RAIN.NO_TYPH_VAL_file_NAME = "NoTyphValDataset.txt"
__C.RAIN.NO_TYPH_TEST_file_NAME = "NoTyphTestDataset.txt"

__C.RAIN.EVALUATION = edict()
__C.RAIN.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 20,30])  # ([0.5, 5, 20, 50])
__C.RAIN.EVALUATION.CENTRAL_REGION = (0, 0, 600, 500)  # (120, 120, 360, 360)
__C.RAIN.EVALUATION.BALANCING_WEIGHTS = (1, 1, 1, 1, 1, 1,1)  # (1, 5, 10, 20, 30)  # (1, 1, 2, 5, 10, 30)

__C.RAIN.BENCHMARK = edict()

__C.RAIN.BENCHMARK.VISUALIZE_SEQ_NUM = 12  # Number of sequences that will be plotted and saved to the benchmark directory
__C.RAIN.BENCHMARK.IN_LEN = 6  # The maximum input length to ensure that all models are tested on the same set of input data
__C.RAIN.BENCHMARK.OUT_LEN = 6  # The maximum output length to ensure that all models are tested on the same set of input data
__C.RAIN.BENCHMARK.STRIDE = 5

__C.MODEL = edict()
from nowcasting.models.model import activation

__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
