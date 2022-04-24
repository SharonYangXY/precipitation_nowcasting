import torch
from experiments.config import cfg
import os.path as osp
import os
import shutil
from tensorboardX import SummaryWriter
from tqdm import tqdm

import numpy as np
from dataset.DataLoaderX import DataLoaderX
from dataset.DataProfetcher import DataPrefetcher
from utils.logger import Logger

log = Logger(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'logs/typh_convLSTM.log'), level='debug')




def test(train_dataset, test_dataset, val_dataset, encoder_forecaster, optimizer, criterion, lr_scheduler,
                   batch_size, epochs, folder_name, evaluater, probToPixel=None, model_path=None, root=None, typh_test_file_name=None):
    
    test_files = open(os.path.join(root, typh_test_file_name)).readlines()

    final_print_file = open(os.path.join(root, "GCN_final_output.txt"), "w")
    

    save_dir = osp.join(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    model_save_dir = osp.join(save_dir, 'predictions')
    if osp.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.mkdir(model_save_dir)
    
    log_dir = osp.join(save_dir, 'test_logs')
    all_scalars_file_name = osp.join(save_dir, "all_scalars.json")

    writer = SummaryWriter(log_dir)
    # val_data_rain, val_data_typhoon, val_label_rain, val_label_typhoon \
    val_loader = DataLoaderX(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        num_workers=8)

    encoder_forecaster.load_state_dict(torch.load(model_path))
    print ("model params loaded......")

    # 开始验证
    log.logger.info("开始验证")
    with torch.no_grad():
        encoder_forecaster.eval()
        valid_loss = 0.0
        val_batch_loss = 0.0

        val_iter = 0
        val_prefetcher = DataPrefetcher(val_loader)
        val_batch = val_prefetcher.next()
        
        print ("test_files and len(val_loader) should be same...", len(test_files), len(val_loader))
        index = 0
        while val_batch is not None:
            val_data_rain, val_data_typhoon, val_label_rain, val_label_typhoon, val_mask, gcn_masks, typh_gcn_masks = val_batch
            output = encoder_forecaster(val_data_rain, gcn_masks, typh_gcn_masks)
            
            final_save_path = test_files[index].strip().split(",")[0].replace("\\", "/").split("/")[-1]
            
            log.logger.info("final_save_path %s " % final_save_path)
            
            save_path = os.path.join(model_save_dir, final_save_path)
            
            loss = criterion(output, val_label_rain, val_mask)
            valid_loss += loss.item()
            val_batch_loss += loss.item()
            valid_label_numpy = val_label_rain.transpose(1, 0).cpu().numpy()  # .transpose(1,0) added by me.
            output_numpy = np.clip(output.transpose(1, 0).detach().cpu().numpy(), 0.0, 300.0)  # as up.....
            
            # save numpy to the file......
            np.savez(save_path, prediction=output_numpy)
            
            evaluater.update(valid_label_numpy, output_numpy, val_mask.transpose(1, 0).cpu().numpy())

            if (val_iter + 1) % 10 == 0:
                log.logger.info(
                    'EAL_Progress: %s iter / %s iters' %
                    (val_iter, len(val_loader)))
                val_batch_loss = 0.0
            val_iter += 1
            val_batch = val_prefetcher.next()
            
            index += 1
            
        valid_balanced_mse, valid_balanced_mae, valid_pod, valid_far, valid_csi, valid_hss, valid_pc, valid_recall, valid_f1_score = evaluater.calculate_stat()
        
        log.logger.info("valid_balanced_mse" % valid_balanced_mse)
        log.logger.info("valid_pod" % valid_pod)
        log.logger.info("valid_far" % valid_far)
        log.logger.info("valid_csi" % valid_csi)
        log.logger.info("valid_hss" % valid_hss)
        log.logger.info("valid_f1_score" % valid_f1_score)
        
        evaluater.clear_all()

        valid_loss = valid_loss / (len(val_loader) * batch_size)

    log.logger.info("test_dataset END")

    writer.export_scalars_to_json(all_scalars_file_name)
    plot_result(writer, 1,
                (valid_balanced_mse, valid_balanced_mae, valid_pod, valid_far, valid_csi, valid_hss, valid_pc,
                 valid_recall, valid_f1_score),
                (valid_balanced_mse, valid_balanced_mae, valid_pod, valid_far, valid_csi, valid_hss, valid_pc,
                 valid_recall, valid_f1_score))

    writer.close()


def plot_result(writer, itera, train_result, valid_result):
    # balanced_mse, balanced_mae, pod, far, csi, hss, pc, recall, f1_score
    train_balanced_mse, train_balanced_mae, train_pod, train_far, train_csi, train_hss, train_pc, train_recall, train_f1_score = train_result
    train_balanced_mse, train_balanced_mae, train_pod, train_far, train_csi, train_hss, train_pc, train_recall, train_f1_score = \
        np.nan_to_num(train_balanced_mse), \
        np.nan_to_num(train_balanced_mae), \
        np.nan_to_num(train_pod), \
        np.nan_to_num(train_far), \
        np.nan_to_num(train_csi), \
        np.nan_to_num(train_hss), \
        np.nan_to_num(train_pc), \
        np.nan_to_num(train_recall), \
        np.nan_to_num(train_f1_score)

    valid_balanced_mse, valid_balanced_mae, valid_pod, valid_far, valid_csi, valid_hss, valid_pc, valid_recall, valid_f1_score = valid_result
    valid_balanced_mse, valid_balanced_mae, valid_pod, valid_far, valid_csi, valid_hss, valid_pc, valid_recall, valid_f1_score = \
        np.nan_to_num(valid_balanced_mse), \
        np.nan_to_num(valid_balanced_mae), \
        np.nan_to_num(valid_pod), \
        np.nan_to_num(valid_far), \
        np.nan_to_num(valid_csi), \
        np.nan_to_num(valid_hss), \
        np.nan_to_num(valid_pc), \
        np.nan_to_num(valid_recall), \
        np.nan_to_num(valid_f1_score)

    writer.add_scalars("balanced_mse", {
        "train": train_balanced_mse.mean(),
        "valid": valid_balanced_mse.mean()
    }, itera)
    
    log.logger.info("balanced_mse mean : %s " % valid_balanced_mse.mean())

    writer.add_scalars("balanced_mae", {
        "train": train_balanced_mae.mean(),
        "valid": valid_balanced_mae.mean()
    }, itera)
    
    log.logger.info("balanced_mae mean : %s " % valid_balanced_mae.mean())

    for i, thresh in enumerate(cfg.RAIN.EVALUATION.THRESHOLDS):
        writer.add_scalars("pod&precision/{}".format(thresh), {
            "train": train_pod[:, i].mean(),
            "valid": valid_pod[:, i].mean(),
        }, itera)
        
#         log.logger.info("pod&precision : %s " % valid_pod[:, i].mean())
        
#         for j in range(train_pod.shape[0]):
#             writer.add_scalars("pod&precision/{}".format(thresh), {
#                 "train_frame" + str(j + 1): train_pod[j, i],
#                 "valid_frame" + str(j + 1): valid_pod[j, i]
#             }, itera)

#     for i, thresh in enumerate(cfg.RAIN.EVALUATION.THRESHOLDS):
#         writer.add_scalars("far/{}".format(thresh), {
#             "train": train_far[:, i].mean(),
#             "valid": valid_far[:, i].mean()
#         }, itera)
#         for j in range(train_far.shape[0]):
#             writer.add_scalars("far/{}".format(thresh), {
#                 "train_frame" + str(j + 1): train_far[j, i],
#                 "valid_frame" + str(j + 1): valid_far[j, i]
#             }, itera)

    for i, thresh in enumerate(cfg.RAIN.EVALUATION.THRESHOLDS):
        writer.add_scalars("csi/{}".format(thresh), {
            "train": train_csi[:, i].mean(),
            "valid": valid_csi[:, i].mean()
        }, itera)
        
        log.logger.info("csi @ %s mean : %s " % (thresh, valid_csi[:, i].mean()))
        print("csi @ mean ", thresh, valid_csi[:, i].mean())
        
        for j in range(train_csi.shape[0]):
            writer.add_scalars("csi/{}".format(thresh), {
                "train_frame" + str(j + 1): train_csi[j, i],
                "valid_frame" + str(j + 1): valid_csi[j, i]
            }, itera)
            
#             log.logger.info("balanced_mse mean : %s " % valid_balanced_mse.mean())

    for i, thresh in enumerate(cfg.RAIN.EVALUATION.THRESHOLDS):
        writer.add_scalars("hss/{}".format(thresh), {
            "train": train_hss[:, i].mean(),
            "valid": valid_hss[:, i].mean()
        }, itera)
        
        log.logger.info("hss @ %s mean : %s " % (thresh, valid_hss[:, i].mean()))
        print("hss @ mean ", thresh, valid_hss[:, i].mean())
        
        for j in range(train_hss.shape[0]):
            writer.add_scalars("hss/{}".format(thresh), {
                "train_frame" + str(j + 1): train_hss[j, i],
                "valid_frame" + str(j + 1): valid_hss[j, i]
            }, itera)


    for i, thresh in enumerate(cfg.RAIN.EVALUATION.THRESHOLDS):
        writer.add_scalars("f1_score/{}".format(thresh), {
            "train": train_f1_score[:, i].mean(),
            "valid": valid_f1_score[:, i].mean()
        }, itera)
        
        log.logger.info("f1_score @ %s mean : %s " % (thresh, valid_f1_score[:, i].mean()))
        
        print("valid_f1_score @ mean ", thresh, valid_f1_score[:, i].mean())
        
        for j in range(train_f1_score.shape[0]):
            writer.add_scalars("f1_score/{}".format(thresh), {
                "train_frame" + str(j + 1): train_f1_score[j, i],
                "valid_frame" + str(j + 1): valid_f1_score[j, i]
            }, itera)


