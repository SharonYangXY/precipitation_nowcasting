import sys

sys.path.insert(0, '../../')
import torch
import torch.nn as nn
from experiments.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.gcn_encoder import Encoder
from nowcasting.models.model import EF_GCN as EF
from torch.optim import lr_scheduler
from nowcasting.evaluation.loss import Weighted_mse_mae
from experiments.gcn_train_test import train_and_test
import os
from experiments.net_params import convlstm_encoder_params, convlstm_forecaster_params
from dataset.PrecipitationTyphAreaDataset import PrecipitationTyphAreaDataset as Dataset
from nowcasting.evaluation.evaluation import GPMEvaluation
import warnings

warnings.filterwarnings('ignore')

### Config

batch_size = cfg.GLOBAL.BATCH_SZIE
epoch = 30

LR_step_size = 3000
gamma = 0.7

LR = 1e-4
WD = 1e-6

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)
# criterion = nn.MSELoss().to(cfg.GLOBAL.DEVICE) #nn.CrossEntropyLoss().to(cfg.GLOBAL.DEVICE)

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster)

# multigpu
encoder_forecaster = nn.DataParallel(encoder_forecaster, device_ids=[0, 1]).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR, weight_decay=WD)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]

OUT_LEN = cfg.RAIN.BENCHMARK.OUT_LEN
evaluater = GPMEvaluation(seq_len=OUT_LEN, use_central=False)

# print(encoder_forecaster)

root = cfg.RAIN.ROOT
# train_file_name = cfg.RAIN.TRAIN_file_NAME
# val_file_name = cfg.RAIN.VAL_file_NAME
# test_file_name = cfg.RAIN.TEST_file_NAME
#
# train_dataset = Dataset(root, train_file_name)
# test_dataset = Dataset(root, test_file_name)
# val_dataset = Dataset(root, val_file_name)

typh_train_file_name = cfg.RAIN.TYPH_TRAIN_file_NAME
typh_val_file_name = cfg.RAIN.TYPH_VAL_file_NAME
typh_test_file_name = cfg.RAIN.TYPH_TEST_file_NAME

typh_train_dataset = Dataset(root, typh_train_file_name)
typh_test_dataset = Dataset(root, typh_test_file_name)
typh_val_dataset = Dataset(root, typh_val_file_name)

# no_typh_train_file_name = cfg.RAIN.NO_TYPH_TRAIN_file_NAME
# no_typh_val_file_name = cfg.RAIN.NO_TYPH_VAL_file_NAME
# no_typh_test_file_name = cfg.RAIN.NO_TYPH_TEST_file_NAME
#
# no_typh_train_dataset = Dataset(root, no_typh_train_file_name)
# no_typh_test_dataset = Dataset(root, no_typh_test_file_name)
# no_typh_val_dataset = Dataset(root, no_typh_val_file_name)

if __name__ == '__main__':
    print(encoder_forecaster)
    # train_and_test(train_dataset,test_dataset,val_dataset,encoder_forecaster, optimizer, criterion, exp_lr_scheduler, batch_size, epoch, folder_name,evaluater)
    train_and_test(typh_train_dataset, typh_test_dataset, typh_val_dataset, encoder_forecaster, optimizer, criterion,
                   exp_lr_scheduler, batch_size, epoch, folder_name, evaluater)
    # train_and_test(no_typh_train_dataset, no_typh_test_dataset, no_typh_val_dataset, encoder_forecaster, optimizer,
    #                criterion, exp_lr_scheduler, batch_size, epoch, folder_name,evaluater)
