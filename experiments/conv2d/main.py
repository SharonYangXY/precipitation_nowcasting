import sys
sys.path.insert(0, '../../')
import torch
from experiments.config import cfg
from torch.optim import lr_scheduler
from nowcasting.evaluation.loss import Weighted_mse_mae
import os
from experiments.net_params import conv2d_params
from nowcasting.models.model import Predictor
from experiments.my_train_test import train_and_test
from dataset.PrecipitationDataset import PrecipitationDataset as Dataset
from nowcasting.evaluation.evaluation import GPMEvaluation
import os

### Config

# batch_size = cfg.GLOBAL.BATCH_SZIE
batch_size = 4
epoch = 30

LR_step_size = 4000
gamma = 0.7

LR = 1e-3 # 1e-3
WD = 1e-6

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)

encoder_forecaster = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)
# multigpu
encoder_forecaster = nn.DataParallel(encoder_forecaster, device_ids=[0, 1]).to(cfg.GLOBAL.DEVICE)

# data = torch.randn(5, 4, 1, 480, 480)
# output = model(data)
# print(output.size())

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR, weight_decay=WD)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)
# mult_step_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30000, 60000], gamma=0.1)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1] + "_no_seed"

OUT_LEN = cfg.RAIN.BENCHMARK.OUT_LEN
evaluater = GPMEvaluation(seq_len=OUT_LEN, use_central=False)

root = cfg.RAIN.ROOT

typh_train_file_name = cfg.RAIN.TYPH_TRAIN_file_NAME
typh_val_file_name = cfg.RAIN.TYPH_VAL_file_NAME
typh_test_file_name = cfg.RAIN.TYPH_TEST_file_NAME

typh_train_dataset = Dataset(root, typh_train_file_name)
typh_test_dataset = Dataset(root, typh_test_file_name)
typh_val_dataset = Dataset(root, typh_val_file_name)

if __name__ == '__main__':
    print(encoder_forecaster)
    train_and_test(typh_train_dataset, typh_test_dataset, typh_val_dataset, encoder_forecaster, optimizer, criterion,
                   exp_lr_scheduler, batch_size, epoch, folder_name, evaluater)
