import sys

sys.path.insert(0, '../../')
import torch
import torch.nn as nn
from experiments.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.typh_encoder import TyphEncoder as Encoder
from nowcasting.models.gcn_encoder import Encoder as base_image_Encoder
from nowcasting.models.fusion_gcn_encoder import FusionGCNEncoder as FusionEncoder
# from nowcasting.models.model import EF
from nowcasting.models.model import EF_GCN as EF
from torch.optim import lr_scheduler
from nowcasting.evaluation.loss import Weighted_mse_mae
# from experiments.my_train_test import train_and_test
from experiments.gcn_train_test import train_and_test
import os
from experiments.net_params import fusion_convlstm_encoder_params, fusion_lstm_params, fusion_convlstm_forecaster_params
from dataset.PrecipitationFusionTyphAreaDataset import PrecipitationFusionTyphAreaDataset as Dataset
from nowcasting.evaluation.evaluation import GPMEvaluation
import warnings
import numpy as np

warnings.filterwarnings('ignore')

### Config

batch_size = cfg.GLOBAL.BATCH_SZIE
epoch = 30

LR_step_size = 4000  # 20000
gamma = 0.7

LR = 1e-3  # 1e-3 1e-2
WD = 1e-6

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic =True
# torch.backends.cudnn.enabled = True


criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)
# criterion = nn.MSELoss().to(cfg.GLOBAL.DEVICE) #nn.CrossEntropyLoss().to(cfg.GLOBAL.DEVICE)


# Here, there has two ways for GCN. One is both existed in image and typh encoder; another is only existed in the image encoder.
# Note it!!!!!!!!!!!!!!!

image_encoder = base_image_Encoder(fusion_convlstm_encoder_params[0], fusion_convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
typh_encoder = Encoder(fusion_lstm_params[0], fusion_lstm_params[1]).to(cfg.GLOBAL.DEVICE)

fusion_encoder = FusionEncoder(image_encoder, typh_encoder)

forecaster = Forecaster(fusion_convlstm_forecaster_params[0], fusion_convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(fusion_encoder, forecaster)

# multigpu
encoder_forecaster = nn.DataParallel(encoder_forecaster, device_ids=[0, 1]).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR, weight_decay=WD)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1] + "_no_seed"

OUT_LEN = cfg.RAIN.BENCHMARK.OUT_LEN
evaluater = GPMEvaluation(seq_len=OUT_LEN, use_central=False)

print(encoder_forecaster)

root = cfg.RAIN.MultiChannelROOT
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
    #     print(encoder_forecaster)
    #     # train_and_test(train_dataset,test_dataset,val_dataset,encoder_forecaster, optimizer, criterion, exp_lr_scheduler, batch_size, epoch, folder_name,evaluater)
    train_and_test(typh_train_dataset, typh_test_dataset, typh_val_dataset, encoder_forecaster, optimizer, criterion,
                   exp_lr_scheduler, batch_size, epoch, folder_name, evaluater)
#     train_and_test(no_typh_train_dataset, no_typh_test_dataset, no_typh_val_dataset, encoder_forecaster, optimizer,
#                    criterion, exp_lr_scheduler, batch_size, epoch, folder_name,evaluater)
