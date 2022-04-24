import torch
import torch.nn as nn
from experiments.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.gcn_encoder import Encoder as base_image_Encoder
from nowcasting.models.gcn_typh_encoder import Encoder as base_typh_Encoder
from nowcasting.models.fusion_gcn_encoder import FusionGCNEncoder as FusionEncoder

from nowcasting.models.model import EF_GCN as EF
from torch.optim import lr_scheduler
from nowcasting.evaluation.loss import Weighted_mse_mae
from experiments.gcn_train_test import train_and_test
import os
from experiments.net_params import fusion_convlstm_encoder_params, fusion_lstm_params, fusion_convlstm_forecaster_params
from dataset.PrecipitationFusionTyphAreaDataset import PrecipitationFusionTyphAreaDataset as Dataset
from nowcasting.evaluation.evaluation import GPMEvaluation
import warnings

warnings.filterwarnings('ignore')

### Config

# batch_size = cfg.GLOBAL.BATCH_SZIE
batch_size = 8
epoch = 300

LR_step_size = 4000  # 20000
gamma = 0.7

LR = 1e-3  # 1e-3 1e-2
WD = 1e-6

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)

image_encoder = base_image_Encoder(fusion_convlstm_encoder_params[0], fusion_convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
typh_encoder = base_typh_Encoder(fusion_lstm_params[0], fusion_lstm_params[1]).to(cfg.GLOBAL.DEVICE)

fusion_encoder = FusionEncoder(image_encoder, typh_encoder)

forecaster = Forecaster(fusion_convlstm_forecaster_params[0], fusion_convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(fusion_encoder, forecaster)

# multigpu
encoder_forecaster = nn.DataParallel(encoder_forecaster, device_ids=[0,1]).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR, weight_decay=WD)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1] + "_no_seed_double_GCN_0505_gcn0_5_lr1e3_S40_1025"

OUT_LEN = cfg.RAIN.BENCHMARK.OUT_LEN
evaluater = GPMEvaluation(seq_len=OUT_LEN, use_central=False)

print(encoder_forecaster)

root = cfg.RAIN.MultiChannelROOT

typh_train_file_name = cfg.RAIN.TYPH_TRAIN_file_NAME
typh_val_file_name = cfg.RAIN.TYPH_VAL_file_NAME
typh_test_file_name = cfg.RAIN.TYPH_TEST_file_NAME

typh_train_dataset = Dataset(root, typh_train_file_name)
typh_test_dataset = Dataset(root, typh_test_file_name)
typh_val_dataset = Dataset(root, typh_val_file_name)


if __name__ == '__main__':
   train_and_test(typh_train_dataset, typh_test_dataset, typh_val_dataset, encoder_forecaster, optimizer, criterion,
                   exp_lr_scheduler, batch_size, epoch, folder_name, evaluater)
