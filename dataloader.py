# -*- coding: utf-8 -*-
import torch
from torch.utils import data
import numpy as np
import SimpleITK as sitk
from Transform import transform_img_lab
import warnings

warnings.filterwarnings("ignore")


def read_file_from_txt(txt_path):
    files = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                files.append(line.strip())
    except FileNotFoundError:
        print(f"Warning: File not found {txt_path}")
    return files


class FedsodaDataloader(data.Dataset):
    def __init__(self, args, client_id=0):
        super(FedsodaDataloader, self).__init__()
        # Use client-specific paths from args if provided, otherwise fallback to index-based naming
        train_image_txt = getattr(args, f'c{client_id}_train_image_txt', args.c_train_image_txt)
        train_label_txt = getattr(args, f'c{client_id}_train_label_txt', args.c_train_label_txt)
        
        self.image_file = read_file_from_txt(train_image_txt)
        self.label_file = read_file_from_txt(train_label_txt)
        self.shape = (args.ROI_shape, args.ROI_shape)
        self.args = args
        self.client_id = client_id

    def __getitem__(self, index):
        image_path = self.image_file[index]
        label_path = self.label_file[index]

        image = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)

        if label.ndim == 3 and label.shape[0] == 3:
            label = label[0]

        image = image.astype(dtype=np.float32)
        label = label.astype(dtype=np.float32)
        label = np.where(label > 0, 1, 0)

        # Load client-specific mean/std
        mean_std_path = getattr(self.args, f'Meanstd_dir{self.client_id if self.client_id > 0 else ""}')
        mean, std = np.load(mean_std_path)
        
        image = (image - mean) / std
        if np.max(label) > 0:
            label = label / np.max(label)

        _, y, x = image.shape
        center_y = np.random.randint(0, y - self.shape[0] + 1)
        center_x = np.random.randint(0, x - self.shape[1] + 1)
        
        image = image[:, center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]
        label = label[center_y:self.shape[0] + center_y, center_x:self.shape[1] + center_x]

        label = label[np.newaxis, :, :]

        # Data Augmentation
        data_dict = transform_img_lab(image, label, self.args)
        image_trans = data_dict['image']
        label_trans = data_dict['label']
        
        if isinstance(image_trans, torch.Tensor):
            image_trans = image_trans.numpy()
        if isinstance(label_trans, torch.Tensor):
            label_trans = label_trans.numpy()

        return image_trans, label_trans

    def __len__(self):
        return len(self.image_file)
