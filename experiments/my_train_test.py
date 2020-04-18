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

# now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# 设置默认的level为DEBUG
# 设置log的格式
# logging.basicConfig(
#     level=logging.INFO,
#     filename=os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR,'logs/typh_convLSTM_' + now + '.log'),
#     filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志#a是追加模式，默认如果不写的话，就是追加模式
#     format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
# )
log = Logger(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'logs/typh_convLSTM.log'), level='debug')


def train_and_test(train_dataset, test_dataset, val_dataset, encoder_forecaster, optimizer, criterion, lr_scheduler,
                   batch_size, epochs, folder_name, evaluater, probToPixel=None):
    train_loss = 0.0
    train_batch_loss = 0.0
    save_dir = osp.join(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    model_save_dir = osp.join(save_dir, 'models')
    log_dir = osp.join(save_dir, 'logs')
    all_scalars_file_name = osp.join(save_dir, "all_scalars.json")
    if osp.exists(all_scalars_file_name):
        os.remove(all_scalars_file_name)
    if osp.exists(log_dir):
        shutil.rmtree(log_dir)
    if osp.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.mkdir(model_save_dir)

    writer = SummaryWriter(log_dir)
    train_loader = DataLoaderX(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=8)
    # val_data_rain, val_data_typhoon, val_label_rain, val_label_typhoon \
    val_loader = DataLoaderX(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        num_workers=8)

    for cur_epoch in tqdm(range(epochs)):
        print("\n")
        log.logger.info("开始epoch % s 训练" % cur_epoch)
        train_prefetcher = DataPrefetcher(train_loader)
        train_batch = train_prefetcher.next()
        iter = 0
        while train_batch is not None:
            train_data_rain, train_data_typhoon, train_label_rain, train_label_typhoon, train_mask = train_batch
            # mask  (mask_batchsize,6,1,600,500)

            lr_scheduler.step()
            encoder_forecaster.train()
            optimizer.zero_grad()
            output = encoder_forecaster(train_data_rain)
            # [2, 6, 1, 600, 500] ==> N, C ,H ,W & after squeeze(2) ===> [2, 6, 600, 500]
            loss = criterion(output, train_label_rain, train_mask)
            loss.backward()

            torch.nn.utils.clip_grad_value_(encoder_forecaster.parameters(), clip_value=50.0)
            optimizer.step()
            train_loss += loss.item()
            train_batch_loss += loss.item()

            train_label_numpy = train_label_rain.transpose(1, 0).cpu().numpy()
            output_numpy = np.clip(output.transpose(1, 0).detach().cpu().numpy(), 0.0, 300.0)
            evaluater.update(train_label_numpy, output_numpy, train_mask.transpose(1, 0).cpu().numpy())
            if (iter + 1) % 10 == 0:
                log.logger.info(
                    'Epoch %s / %s  Train_Progress: %s iter / %s iters, train_batch_loss: %s' %
                    (cur_epoch, epochs, iter, len(train_loader), train_batch_loss / (10 * batch_size)))
                train_batch_loss = 0.0
            iter += 1
            train_batch = train_prefetcher.next()

        log.logger.info("计算epoch % s 训练loss & evaluation" % cur_epoch)

        # train_pod, train_far, train_csi, train_hss, train_gss, train_mse, train_mae, train_balanced_mse, train_balanced_mae, train_gdl = evaluater.calculate_stat()
        train_balanced_mse, train_balanced_mae, train_pod, train_far, train_csi, train_hss, train_pc, train_recall, train_f1_score = evaluater.calculate_stat()
        test_iteration_interval = len(train_loader) * batch_size
        train_loss = train_loss / test_iteration_interval
        evaluater.clear_all()
        log.logger.info("epoch % s 训练结束" % cur_epoch)

        # 开始验证
        log.logger.info("开始epoch % s 验证" % cur_epoch)
        with torch.no_grad():
            encoder_forecaster.eval()
            valid_loss = 0.0
            val_batch_loss = 0.0

            val_iter = 0
            val_prefetcher = DataPrefetcher(val_loader)
            val_batch = val_prefetcher.next()

            while val_batch is not None:
                val_data_rain, val_data_typhoon, val_label_rain, val_label_typhoon, val_mask = val_batch
                output = encoder_forecaster(val_data_rain)
                loss = criterion(output, val_label_rain, val_mask)
                valid_loss += loss.item()
                val_batch_loss += loss.item()
                valid_label_numpy = val_label_rain.transpose(1, 0).cpu().numpy()  # .transpose(1,0) added by me.
                output_numpy = np.clip(output.transpose(1, 0).detach().cpu().numpy(), 0.0, 300.0)  # as up.....
                evaluater.update(valid_label_numpy, output_numpy, val_mask.transpose(1, 0).cpu().numpy())

                if (val_iter + 1) % 10 == 0:
                    log.logger.info(
                        'Epoch %s / %s  EAL_Progress: %s iter / %s iters, val_batch_loss: %s' %
                        (cur_epoch, epochs, val_iter, len(val_loader), val_batch_loss / (10 * batch_size)))
                    val_batch_loss = 0.0
                val_iter += 1
                val_batch = val_prefetcher.next()


            # valid_pod, valid_far, valid_csi, valid_hss, valid_gss, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae, valid_gdl = evaluater.calculate_stat()
            valid_balanced_mse, valid_balanced_mae, valid_pod, valid_far, valid_csi, valid_hss, valid_pc, valid_recall, valid_f1_score = evaluater.calculate_stat()
            evaluater.clear_all()

            log.logger.info("计算epoch % s 验证loss" % cur_epoch)
            valid_loss = valid_loss / (len(val_loader) * batch_size)
            log.logger.info("epoch % s 验证结束" % cur_epoch)

        log.logger.info("记录epoch % s 训练以及验证loss" % cur_epoch)
        log.logger.info('Evaluation of Curr_epoch %s / total_epoch %s : train_loss: %s , valid_loss: %s' % (
            cur_epoch, epochs, train_loss, valid_loss))
        writer.add_scalars("loss", {
            "train": train_loss,
            "valid": valid_loss
        }, cur_epoch)

        log.logger.info("epoch % s END" % cur_epoch)
        train_loss = 0.0

        writer.export_scalars_to_json(all_scalars_file_name)
        plot_result(writer, cur_epoch,
                    (train_balanced_mse, train_balanced_mae, train_pod, train_far, train_csi, train_hss, train_pc, train_recall, train_f1_score),
                    (valid_balanced_mse, valid_balanced_mae, valid_pod, valid_far, valid_csi, valid_hss, valid_pc, valid_recall, valid_f1_score))

        # 保存模型
        log.logger.info("保存epoch % s 模型结果" % cur_epoch)
        if 1:
            torch.save(encoder_forecaster.state_dict(),
                       osp.join(model_save_dir, 'typh_encoder_forecaster_{}.pth'.format(cur_epoch)))

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

    writer.add_scalars("balanced_mae", {
        "train": train_balanced_mae.mean(),
        "valid": valid_balanced_mae.mean()
    }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        writer.add_scalars("pod&precision/{}".format(thresh), {
            "train": train_pod[:, i].mean(),
            "valid": valid_pod[:, i].mean(),
        }, itera)
        for j in range(train_pod.shape[0]):
            writer.add_scalars("pod&precision/{}".format(thresh), {
                "train_frame" + str(j + 1): train_pod[j, i],
                "valid_frame" + str(j + 1): valid_pod[j, i]
            }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        writer.add_scalars("far/{}".format(thresh), {
            "train": train_far[:, i].mean(),
            "valid": valid_far[:, i].mean()
        }, itera)
        for j in range(train_far.shape[0]):
            writer.add_scalars("far/{}".format(thresh), {
                "train_frame" + str(j + 1): train_far[j, i],
                "valid_frame" + str(j + 1): valid_far[j, i]
            }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        writer.add_scalars("csi/{}".format(thresh), {
            "train": train_csi[:, i].mean(),
            "valid": valid_csi[:, i].mean()
        }, itera)
        for j in range(train_csi.shape[0]):
            writer.add_scalars("csi/{}".format(thresh), {
                "train_frame" + str(j + 1): train_csi[j, i],
                "valid_frame" + str(j + 1): valid_csi[j, i]
            }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        writer.add_scalars("hss/{}".format(thresh), {
            "train": train_hss[:, i].mean(),
            "valid": valid_hss[:, i].mean()
        }, itera)
        for j in range(train_hss.shape[0]):
            writer.add_scalars("hss/{}".format(thresh), {
                "train_frame" + str(j + 1): train_hss[j, i],
                "valid_frame" + str(j + 1): valid_hss[j, i]
            }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        writer.add_scalars("pc/{}".format(thresh), {
            "train": train_pc[:, i].mean(),
            "valid": valid_pc[:, i].mean()
        }, itera)
        for j in range(train_pc.shape[0]):
            writer.add_scalars("pc/{}".format(thresh), {
                "train_frame" + str(j + 1): train_pc[j, i],
                "valid_frame" + str(j + 1): valid_pc[j, i]
            }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        writer.add_scalars("recall/{}".format(thresh), {
            "train": train_recall[:, i].mean(),
            "valid": valid_recall[:, i].mean()
        }, itera)
        for j in range(train_recall.shape[0]):
            writer.add_scalars("recall/{}".format(thresh), {
                "train_frame" + str(j + 1): train_recall[j, i],
                "valid_frame" + str(j + 1): valid_recall[j, i]
            }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        writer.add_scalars("f1_score/{}".format(thresh), {
            "train": train_f1_score[:, i].mean(),
            "valid": valid_f1_score[:, i].mean()
        }, itera)
        for j in range(train_f1_score.shape[0]):
            writer.add_scalars("f1_score/{}".format(thresh), {
                "train_frame" + str(j + 1): train_f1_score[j, i],
                "valid_frame" + str(j + 1): valid_f1_score[j, i]
            }, itera)
