# -*- coding: utf-8 -*-
import argparse
import os
from trainer import Process

def get_args():
    parser = argparse.ArgumentParser(description='Fedsoda: Federated Learning for Medical Image Segmentation')

    # General options
    parser.add_argument('--client_id', type=int, default=0, help='Client ID (0-6)')
    parser.add_argument('--base_dir', type=str, default='./Fedsoda_Project/', help='Base directory for all outputs')
    parser.add_argument('--data_dir', type=str, default='./Data/Cell/', help='Directory containing mean/std files')
    parser.add_argument('--txt_dir', type=str, default='./Txt/Txt_Cell/', help='Directory containing txt files for data listing')

    # Network options
    parser.add_argument('--n_channels', default=3, type=int, help='input channels')
    parser.add_argument('--n_classes', default=1, type=int, help='output channels')
    parser.add_argument('--n_basic_layer', default=32, type=int, help='basic layer numbers')
    parser.add_argument('--model_name', default='Fedsoda_Model', help='Weights name')
    parser.add_argument('--model_name_max', default='Fedsoda_Model_max', help='Max Weights name')
    parser.add_argument('--log_name', default='Fedsoda.log', help='Log name')

    # Training options
    parser.add_argument('--GPU_id', default='0', help='GPU ID')
    parser.add_argument('--ROI_shape', default=512, type=int, help='roi size')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--start_train_step', default=0, type=int, help='Start training step')
    parser.add_argument('--start_verify_step', default=15, type=int, help='Start verifying step')
    parser.add_argument('--n_epoch_per_step', default=5, type=int, help='N epoch per step')
    parser.add_argument('--n_epochs', default=300, type=int, help='Epoch Num')
    parser.add_argument('--if_retrain', default=True, type=bool, help='If Retrain')
    parser.add_argument('--if_onlytest', default=False, type=bool, help='If Only Test')

    args, unknown = parser.parse_known_args()

    # Dynamic Path Setup based on base_dir
    args.checkpoint_dir = os.path.join(args.base_dir, 'Weights/Fedsoda_Centre/')
    args.checkpoint_dir_s = os.path.join(args.checkpoint_dir, 'Server/')
    args.log_dir = os.path.join(args.base_dir, 'Log/Fedsoda_Centre/')
    args.save = os.path.join(args.base_dir, 'Results/Fedsoda_Centre/')
    args.save_path_array = os.path.join(args.base_dir, 'Array/Fedsoda_Centre/')

    # Client-specific paths
    for i in range(7):
        setattr(args, f'checkpoint_dir_c{i}', os.path.join(args.checkpoint_dir, f'C{i}/'))
        setattr(args, f'save_path_c{i}', os.path.join(args.save, f'C{i}/'))
        setattr(args, f'save_path_max_c{i}', os.path.join(args.save, f'C{i}_max/'))
        
        setattr(args, f'c{i}_train_image_txt', os.path.join(args.txt_dir, f'c{i}_train_image.txt'))
        setattr(args, f'c{i}_train_label_txt', os.path.join(args.txt_dir, f'c{i}_train_label.txt'))
        setattr(args, f'c{i}_test_image_txt', os.path.join(args.txt_dir, f'c{i}_test_image.txt'))
        setattr(args, f'c{i}_test_label_txt', os.path.join(args.txt_dir, f'c{i}_test_label.txt'))

    # Meanstd paths (mapping provided names to C0-C6)
    datasets = ['CoNSeP', 'Cpm17', 'CRAG', 'CryoNuSeg', 'Glas', 'Kumar', 'TNBC']
    for i, ds in enumerate(datasets):
        suffix = str(i) if i > 0 else ""
        setattr(args, f'Meanstd_dir{suffix}', os.path.join(args.data_dir, f'{ds}_nii_meanstd.npy'))

    # Legacy attributes for compatibility with generic dataloader
    args.c_train_image_txt = getattr(args, f'c{args.client_id}_train_image_txt')
    args.c_train_label_txt = getattr(args, f'c{args.client_id}_train_label_txt')
    args.Meanstd_dir = getattr(args, f'Meanstd_dir{args.client_id if args.client_id > 0 else ""}')

    return args

if __name__ == '__main__':
    args = get_args()
    Process(args)
