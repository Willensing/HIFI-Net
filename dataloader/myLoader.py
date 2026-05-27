import numpy as np
import cv2
import torch
from torch.utils import data
from torchvision import transforms
import os

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels, edge_labels, cfg):
        self.list_IDs = list_IDs
        self.labels = labels
        self.edge_labels = edge_labels
        self.cfg = cfg
        self.normalize = transforms.Normalize(cfg['dataset_params']['mean'], cfg['dataset_params']['std'])
        self.file_names = list(self.list_IDs.keys())

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = self.list_IDs[file_name]
        mask_path = self.labels[file_name]
        edge_mask_path = self.edge_labels[file_name]

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if mask_path is None or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        if edge_mask_path is None or not os.path.exists(edge_mask_path):
            raise FileNotFoundError(f"Edge mask file not found: {edge_mask_path}")

        image = cv2.imread(image_path, 1)
        mask = cv2.imread(mask_path, 0)
        edge_mask = cv2.imread(edge_mask_path, 0)

        if image is None or mask is None or edge_mask is None:
            raise Exception(f"读取失败：{file_name}")

        im_size = self.cfg['dataset_params']['im_size']
        image = cv2.resize(image, (im_size, im_size))
        image = image / 255.0
        image = np.moveaxis(image, 2, 0)
        image = torch.from_numpy(np.float32(image))
        image = self.normalize(image)

        mask = cv2.resize(mask, (im_size, im_size), interpolation=cv2.INTER_NEAREST) / 255.0
        edge_mask = cv2.resize(edge_mask, (im_size, im_size), interpolation=cv2.INTER_NEAREST) / 255.0

        mask = torch.from_numpy(mask)        # ✅ 不加 unsqueeze
        edge_mask = torch.from_numpy(edge_mask)

        return image, mask, edge_mask


def get_file_names(cfg):
    train_img_dir = cfg['dataset_params']['train_img_dir']
    train_mask_dir = cfg['dataset_params']['train_mask_dir']
    train_edge_dir = cfg['dataset_params']['train_edge_dir']

    val_img_dir = cfg['dataset_params']['val_img_dir']
    val_mask_dir = cfg['dataset_params']['val_mask_dir']
    val_edge_dir = cfg['dataset_params']['val_edge_dir']

    mask_suffixes = ['png', 'PNG', 'tif', 'TIF', 'jpg', 'JPG']

    def find_mask(file_name, folder):
        base = os.path.splitext(file_name)[0]
        for ext in mask_suffixes:
            candidate = os.path.join(folder, f"{base}.{ext}")
            if os.path.exists(candidate):
                return candidate
        print(f"⚠️ Mask not found for {file_name}")
        return None

    def find_edge(file_name, folder):
        base = os.path.splitext(file_name)[0]
        for ext in mask_suffixes:
            candidate = os.path.join(folder, f"{base}.{ext}")
            if os.path.exists(candidate):
                return candidate
        print(f"⚠️ Edge mask not found for {file_name}")
        return None

    train_files = sorted([f for f in os.listdir(train_img_dir) if f.startswith("img")])
    val_files = sorted([f for f in os.listdir(val_img_dir) if f.startswith("img")])

    train_IDs = {f: os.path.join(train_img_dir, f) for f in train_files}
    mask_train_IDs = {f: find_mask(f, train_mask_dir) for f in train_files}
    edge_train_IDs = {f: find_edge(f, train_edge_dir) for f in train_files}

    val_IDs = {f: os.path.join(val_img_dir, f) for f in val_files}
    mask_val_IDs = {f: find_mask(f, val_mask_dir) for f in val_files}
    edge_val_IDs = {f: find_edge(f, val_edge_dir) for f in val_files}

    for name, mask_dict, edge_dict in [
        ("train", mask_train_IDs, edge_train_IDs),
        ("val", mask_val_IDs, edge_val_IDs)
    ]:
        missing_mask = [k for k, v in mask_dict.items() if v is None]
        missing_edge = [k for k, v in edge_dict.items() if v is None]
        if missing_mask:
            print(f"[{name.upper()}] Missing masks: {missing_mask}")
        if missing_edge:
            print(f"[{name.upper()}] Missing edge masks: {missing_edge}")

    return train_IDs, mask_train_IDs, edge_train_IDs, val_IDs, mask_val_IDs, edge_val_IDs


class generator():
    def __init__(self, cfg):
        self.cfg = cfg
        try:
            (self.train_IDs, self.mask_train_IDs, self.edge_train_IDs,
             self.val_IDs, self.mask_val_IDs, self.edge_val_IDs) = get_file_names(cfg)
        except Exception as e:
            print(f"Error loading file paths: {e}")
            raise

    def get_train_generator(self):
        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True, 'num_workers': 4}
        dataset = Dataset(self.train_IDs, self.mask_train_IDs, self.edge_train_IDs, self.cfg)
        return data.DataLoader(dataset, **params)

    def get_val_generator(self):
        batch_size = self.cfg['dataset_params']['batch_size']
        params = {'batch_size': batch_size, 'shuffle': False, 'pin_memory': True, 'num_workers': 4}
        dataset = Dataset(self.val_IDs, self.mask_val_IDs, self.edge_val_IDs, self.cfg)
        return data.DataLoader(dataset, **params)
