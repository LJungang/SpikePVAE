# spikedataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import os
import numpy as np
from natsort import natsorted
from torchvision import transforms

class SpikeDataset(Dataset):
    def __init__(self, img_path, spike_path, transforms=None, data_type='train', neurons_nums=100, spike_times=100):
        self.img_path = Path(img_path)
        self.spike_path = spike_path
        self.transforms = transforms
        self.neurons_nums = neurons_nums
        self.spike_times = spike_times
        self.spikelength = neurons_nums * spike_times
        self.ids = natsorted([f.name for f in self.img_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}])
        self.nums = len(self.ids)
        self.spikedatas = np.load(spike_path)['arr_0'][:]
        self.spikedatas = self.spikedatas.reshape(self.nums, self.spikelength)

        if data_type != 'test':
            # 按顺序划分训练集和验证集
            split_point = int(self.nums * 0.8)  # 80% 为训练集
            if data_type == 'train':
                self.spikedatas = self.spikedatas[:split_point]
                self.ids = self.ids[:split_point]
            elif data_type == 'val':
                self.spikedatas = self.spikedatas[split_point:]
                self.ids = self.ids[split_point:]
        # 如果是 'test'，使用所有数据，不做任何处理

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.ids[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        spike_data = torch.Tensor(self.spikedatas[idx])
        spike_data = spike_data.view(self.neurons_nums, self.spike_times) # (neurons_num, spike_times)
        label = os.path.basename(img_name)[0]
        return spike_data, image # (neurons_num, spike_times), (C, H, W)


def get_dataloaders(img_path, spike_path, batch_size, input_sz, neurons_nums, spike_times, num_workers=4, pin_memory=True):
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((input_sz, input_sz)),
        transforms.ToTensor()
    ])

    # 创建数据集
    train_dataset = SpikeDataset(img_path, spike_path, transform, data_type='train', neurons_nums=neurons_nums, spike_times=spike_times)
    val_dataset = SpikeDataset(img_path, spike_path, transform, data_type='val',neurons_nums=neurons_nums, spike_times=spike_times)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader