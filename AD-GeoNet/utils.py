import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm  # 增加进度条，方便查看预加载进度


class AirfoilDataset(Dataset):
    def __init__(self, dat_dir, img_dir, img_size=224, preload=True, device='cpu'):
        """
        加载分割后的数据集（支持预加载）
        :param dat_dir: 坐标文件目录（如 ../data/split_airfoil/train_uiuc）
        :param img_dir: 图像文件目录（如 ../data/split_airfoil/train_maps）
        :param preload: 是否预加载数据到内存/GPU
        :param device: 预加载设备（cpu/cuda）
        """
        self.data_dir = dat_dir
        self.img_dir = img_dir
        self.img_size = img_size
        self.preload = preload
        self.device = device

        # 1. 收集所有有效文件对（img+dat）
        self.img_files = []
        self.dat_paths = []
        for f in os.listdir(self.img_dir):
            if f.endswith(".png"):
                dat_name = os.path.splitext(f)[0] + ".dat"
                dat_path = os.path.join(dat_dir, dat_name)
                if os.path.exists(dat_path):
                    self.img_files.append(f)
                    self.dat_paths.append(dat_path)

        # 2. 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 3. 预加载所有数据到内存/GPU
        self.preloaded_data = []
        if self.preload:
            print(f"开始预加载数据集（共{len(self.img_files)}个样本）到{self.device}...")
            for idx in tqdm(range(len(self.img_files)), desc="预加载进度"):
                img_tensor, y_tensor, img_name = self._load_single_sample(idx)
                # 转移到指定设备并缓存
                self.preloaded_data.append((
                    img_tensor.to(self.device, non_blocking=True),
                    y_tensor.to(self.device, non_blocking=True),
                    img_name
                ))
            print("数据集预加载完成！")

    def _load_single_sample(self, idx):
        """辅助函数：加载单个样本"""
        # 读取图像
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)

        # 读取dat文件
        dat_path = self.dat_paths[idx]
        y_coords = []
        with open(dat_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    y_coords.append(float(parts[1]))
        y_tensor = torch.tensor(y_coords, dtype=torch.float32)

        return img_tensor, y_tensor, img_name

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if self.preload:
            return self.preloaded_data[idx]
        else:
            return self._load_single_sample(idx)