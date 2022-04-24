import torch
import torch.nn as nn
from experiments.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.evaluation.loss import Weighted_mse_mae
import os
import warnings
from experiments.net_params import convlstm_multiChannel_encoder_params, convlstm_multiChannel_forecaster_params
from dataset.PrecipitationTyphDataset import PrecipitationTyphDataset as Dataset
from experiments.my_test import test
from nowcasting.evaluation.evaluation import GPMEvaluation

warnings.filterwarnings('ignore')

### Config

batch_size = 1 #cfg.GLOBAL.BATCH_SZIE
epoch = 30

LR_step_size = 4000  # 20000
gamma = 0.7

LR = 1e-3  # 1e-3 1e-2
WD = 1e-6


criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)

encoder = Encoder(convlstm_multiChannel_encoder_params[0], convlstm_multiChannel_encoder_params[1]).to(
    cfg.GLOBAL.DEVICE)

forecaster = Forecaster(convlstm_multiChannel_forecaster_params[0], convlstm_multiChannel_forecaster_params[1]).to(
    cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster)

# multigpu
encoder_forecaster = nn.DataParallel(encoder_forecaster, device_ids=[0, 1]).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR, weight_decay=WD)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]+"_sssss"
model_path = '/home/sunpeng/data/yiyi/save/convLSTM_multiChannels__0620/models/typh_encoder_forecaster_10.pth'
OUT_LEN = cfg.RAIN.BENCHMARK.OUT_LEN
evaluater = GPMEvaluation(seq_len=OUT_LEN, use_central=False)

root = cfg.RAIN.MultiChannelROOT

typh_train_file_name = cfg.RAIN.TYPH_TRAIN_file_NAME
typh_val_file_name = cfg.RAIN.TYPH_VAL_file_NAME
typh_test_file_name = cfg.RAIN.TYPH_TEST_file_NAME

typh_train_dataset = Dataset(root, typh_train_file_name)
typh_test_dataset = Dataset(root, typh_test_file_name)
typh_val_dataset = Dataset(root, typh_val_file_name)


if __name__ == '__main__':
    print(encoder_forecaster)

    test(typh_train_dataset, typh_test_dataset, typh_val_dataset, encoder_forecaster, optimizer, criterion,
                   exp_lr_scheduler, batch_size, epoch, folder_name, evaluater, None, model_path, root, typh_test_file_name)

