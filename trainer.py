# -*- coding: utf-8 -*-
import os
import copy
import time
import torch
import logging
import numpy as np
from os.path import join
import SimpleITK as sitk
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Network import UNet
from Loss import cross_loss, balance_bce_loss
from dataloader import FedsodaDataloader
from Params_fusion import Fuse

import warnings
warnings.filterwarnings("ignore")

import random

"""
Fedsoda Training & Orchestration Module
This module handles the local training of clients and the global 
synchronization (federated step). It uses a file-based lock mechanism
where clients wait for all others to finish their local training 
before triggering the fusion logic.
"""

def set_seed(seed=3407):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, server, loader, optimizer, criterion, criterion1, step, epoch, n_epochs, args, model_name):
    """Runs a single epoch of local training."""
    losses = AverageMeter()

    server.eval()
    model.train()
    for batch_idx, (image, label) in enumerate(loader):
        if torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        model.zero_grad()

        output = model(image)

        if step > 0:
            # Distillation-style loss: mimic the global server model
            server_path = os.path.join(args.checkpoint_dir_s, model_name)
            if os.path.exists(server_path):
                server.load_state_dict(torch.load(server_path))
            output1 = server(image)

            loss = criterion(label, output) + criterion1(output1, output)
            losses.update(loss.data, label.size(0))

            loss.backward()
            optimizer.step()
        else:
            # First step: standard supervised loss
            loss = criterion(label, output)
            losses.update(loss.data, label.size(0))

            loss.backward()
            optimizer.step()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                             'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                             'Lr: [%f]' % (optimizer.param_groups[0]['lr']),
                             'Loss %f' % (losses.avg)])
            print(res)

    return losses.avg


def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def Train_net(net, args, model_name, logger, client_id, step=0, per_epoch=5):
    global_net = copy.deepcopy(net)
    client_checkpoint_dir = getattr(args, f'checkpoint_dir_c{client_id}')
    client_model_path = os.path.join(client_checkpoint_dir, model_name)
    
    if os.path.exists(client_model_path):
        net.load_state_dict(torch.load(client_model_path))
        print(f"Loaded client {client_id} model from {client_model_path}")
    
    if torch.cuda.is_available():
        net = net.cuda()
    
    train_dataset = FedsodaDataloader(args, client_id=client_id)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.95))
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    criterion = cross_loss()
    criterion1 = balance_bce_loss()

    logger.info('Client {} start training step{}!'.format(client_id, step))
    for epoch in range(step*per_epoch, (step+1)*per_epoch):
        loss = train_epoch(net, global_net, train_dataloader, optimizer, criterion, criterion1, step, epoch, (step+1)*per_epoch, args, model_name)
        torch.save(net.state_dict(), client_model_path)
        logger.info('Epoch:[{}/{}]  lr={:.6f}  loss={:.5f}'.format(epoch, (step+1)*per_epoch, optimizer.param_groups[0]['lr'], loss))


def read_file_from_txt(txt_path):
    files = []
    with open(txt_path, 'r') as f:
        for line in f:
            files.append(line.strip())
    return files


def reshape_img(image, y, x):
    out = np.zeros([3, y, x], dtype=np.float32)
    out[:, 0:image.shape[1], 0:image.shape[2]] = image[:, 0:image.shape[1], 0:image.shape[2]]
    return out


def predict(model, image_txt, save_path, args, client_id):
    print(f"Predicting test data for client {client_id}")
    model.eval()
    file = read_file_from_txt(image_txt)
    
    mean_std_path = getattr(args, f'Meanstd_dir{client_id if client_id > 0 else ""}')
    mean, std = np.load(mean_std_path)

    for t in range(len(file)):
        image_path = file[t]
        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image).astype(np.float32)

        name = os.path.basename(image_path)
        image = (image - mean) / std
        _, y, x = image.shape

        ind = (max(y, x) // 512 + 1) * 512
        image = reshape_img(image, ind, ind)
        
        predict_map = np.zeros([1, args.n_classes, ind, ind], dtype=np.float32)
        n_map = np.zeros([1, args.n_classes, ind, ind], dtype=np.float32)

        shape = (args.ROI_shape, args.ROI_shape)
        yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        map_kernel = 1 / ((yy - shape[0] // 2) ** 4 + (xx - shape[1] // 2) ** 4 + 1e-8)
        map_kernel = map_kernel[np.newaxis, np.newaxis, :, :]

        image = image[np.newaxis, :, :, :]
        stride_x, stride_y = shape[0] // 2, shape[1] // 2
        
        for i in range(ind // stride_x - 1):
            for j in range(ind // stride_y - 1):
                y_s, y_e = i * stride_x, i * stride_x + shape[0]
                x_s, x_e = j * stride_y, j * stride_y + shape[1]
                
                if y_e > ind or x_e > ind: continue
                
                image_i = torch.from_numpy(image[:, :, y_s:y_e, x_s:x_e])
                if torch.cuda.is_available(): image_i = image_i.cuda()
                output = model(image_i).data.cpu().numpy()

                predict_map[:, :, y_s:y_e, x_s:x_e] += output * map_kernel
                n_map[:, :, y_s:y_e, x_s:x_e] += map_kernel

        output = predict_map / (n_map + 1e-8)
        output_final = output[0, 0, 0:y, 0:x]
        out_img = sitk.GetImageFromArray(output_final)
        sitk.WriteImage(out_img, os.path.join(save_path, name))
    print("finish prediction!")


def Dice(label_txt, pred_dir):
    label_file = read_file_from_txt(label_txt)
    dice_list = []

    for image_path in label_file:
        name = os.path.basename(image_path)
        pred_path = join(pred_dir, name)
        if not os.path.exists(pred_path): continue

        predict = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).astype(np.float32)
        predict = np.where(predict > 0.5, 1, 0)

        groundtruth = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        if groundtruth.ndim == 3 and groundtruth.shape[0] == 3:
            groundtruth = groundtruth[0]
        groundtruth = groundtruth.astype(np.float32)
        groundtruth = np.where(groundtruth > 0, 1, 0)

        if np.max(groundtruth) > 0:
            groundtruth = groundtruth / np.max(groundtruth)

        tmp = predict + groundtruth
        intersection = np.sum(tmp == 2)
        sum_val = np.sum(predict) + np.sum(groundtruth)
        
        dice = (2 * intersection) / (sum_val + 1e-8)
        dice_list.append(dice)
        print(f"{name}: {dice:.4f}")
        
    return np.array(dice_list)


def Create_files(args):
    dirs = [args.checkpoint_dir, args.log_dir, args.checkpoint_dir_s, args.save, args.save_path_array]
    for i in range(7):
        dirs.append(getattr(args, f'checkpoint_dir_c{i}'))
        dirs.append(getattr(args, f'save_path_c{i}'))
        dirs.append(getattr(args, f'save_path_max_c{i}'))
    
    for d in dirs:
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def Predict_Network(net, args, client_id):
    if torch.cuda.is_available():
        net = net.cuda()
    
    client_checkpoint_dir = getattr(args, f'checkpoint_dir_c{client_id}')
    save_path_max = getattr(args, f'save_path_max_c{client_id}')
    test_image_txt = getattr(args, f'c{client_id}_test_image_txt')
    test_label_txt = getattr(args, f'c{client_id}_test_label_txt')

    try:
        net.load_state_dict(torch.load(os.path.join(client_checkpoint_dir, args.model_name_max)))
    except:
        print('Warning: No max weights, using current weights')
        net.load_state_dict(torch.load(os.path.join(client_checkpoint_dir, args.model_name)))
    
    predict(net, test_image_txt, save_path_max, args, client_id)
    dice = Dice(test_label_txt, save_path_max)
    print(f"Client {client_id} Mean Dice: {np.mean(dice):.4f}")


def Process_FL(net, args, client_id):
    dt = datetime.today()
    log_name = f"{dt.date()}_{dt.hour}_{dt.minute}_{dt.second}_C{client_id}_{args.log_name}"
    logger = Get_logger(os.path.join(args.log_dir, log_name))
    logger.info(f'Client {client_id} start training!')

    model_name = args.model_name
    client_checkpoint_dir = getattr(args, f'checkpoint_dir_c{client_id}')
    save_path = getattr(args, f'save_path_c{client_id}')
    test_image_txt = getattr(args, f'c{client_id}_test_image_txt')
    test_label_txt = getattr(args, f'c{client_id}_test_label_txt')

    if not args.if_retrain and os.path.exists(os.path.join(client_checkpoint_dir, model_name)):
        net.load_state_dict(torch.load(os.path.join(client_checkpoint_dir, model_name)))

    # Warm up
    Train_net(net, args, model_name, logger, client_id, step=0, per_epoch=5)

    dice_m = 0
    num_steps = args.n_epochs // args.n_epoch_per_step
    for step in range(args.start_train_step, num_steps):
        step_model_name = f"{args.model_name}_step{step}"
        torch.save(net.state_dict(), os.path.join(client_checkpoint_dir, step_model_name))
        update_model_name = f"{args.model_name}_update{step}"

        # Wait for all clients to finish this step (file-based sync)
        if client_id == 0: # Let client 0 handle fusion logic if desired, or all clients wait
             # In this implementation, each client waits for others
             pass
        
        client_models = [os.path.join(getattr(args, f'checkpoint_dir_c{i}'), step_model_name) for i in range(7)]
        
        count = 0
        input_param_list = [f'Server/{step_model_name}'] + [f'C{i}/{step_model_name}' for i in range(7)]
        output_param_list = [f'Server/{update_model_name}'] + [f'C{i}/{update_model_name}' for i in range(7)]

        while True:
            if all(os.path.exists(m) for m in client_models):
                print(f"Step {step}: All client models ready. Fusing...")
                # Only one process should perform the fusion if they share the same filesystem
                # Or they all do it (it's deterministic)
                Fuse(args.checkpoint_dir, input_param_list, output_param_list, args, net)
                break
            else:
                if count % 12 == 0: print(f"Waiting for other clients (step {step})...")
                time.sleep(5)
                count += 1
        
        Train_net(net, args, update_model_name, logger, client_id, step, args.n_epoch_per_step)

        if step >= args.start_verify_step:
            net.load_state_dict(torch.load(os.path.join(client_checkpoint_dir, update_model_name)))
            predict(net, test_image_txt, save_path, args, client_id)
            dice = Dice(test_label_txt, save_path)
            dice_mean = np.mean(dice)
            if dice_mean > dice_m:
                dice_m = dice_mean
                torch.save(net.state_dict(), os.path.join(client_checkpoint_dir, args.model_name_max))
            logger.info(f'Step {step}: dice_mean={dice_mean:.4f} max_dice={dice_m:.4f}')
            
    logger.info('finish training!')


def Process(args):
    set_seed(3407)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    net = UNet(args)
    Create_files(args)
    
    client_id = args.client_id
    if not args.if_onlytest:
        Process_FL(net, args, client_id)
        Predict_Network(net, args, client_id)
    else:
        Predict_Network(net, args, client_id)
