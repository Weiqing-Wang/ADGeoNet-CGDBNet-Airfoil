import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.encoder.airfoil_vit import AirfoilViT
from src.encoder.utils import AirfoilDataset
def set_all_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_all_seeds(seed=0)
def split_and_save_dataset(
        src_dat_dir="./data/uiuc_convert",
        src_img_dir="./data/airfoil_maps",
        dst_root="./data/split_airfoil",
        val_split=0.1,
        random_seed=0
):

    dir_configs = [
        ("train_maps", src_img_dir),
        ("val_maps", src_img_dir),
        ("train_uiuc", src_dat_dir),
        ("val_uiuc", src_dat_dir)
    ]
    for dir_name, _ in dir_configs:
        dir_path = os.path.join(dst_root, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    samples = []
    for img_file in os.listdir(src_img_dir):
        if not img_file.lower().endswith(".png"):
            continue
        base_name = os.path.splitext(img_file)[0]
        dat_name = f"{base_name}.dat"
        dat_path = os.path.join(src_dat_dir, dat_name)
        if os.path.exists(dat_path):
            samples.append({
                "base_name": base_name,
                "img_path": os.path.join(src_img_dir, img_file),
                "dat_path": dat_path
            })

    np.random.seed(random_seed)
    shuffled_samples = np.random.permutation(samples)
    total = len(shuffled_samples)
    val_size = int(total * val_split)
    train_size = total - val_size
    train_samples = shuffled_samples[:train_size]
    val_samples = shuffled_samples[train_size:]

    def copy_files(sample_list, dataset_type):
        for sample in sample_list:
            dst_img_path = os.path.join(dst_root, f"{dataset_type}_maps",
                                        f"{sample['base_name']}.png")
            shutil.copy2(sample["img_path"], dst_img_path)
            dst_dat_path = os.path.join(dst_root, f"{dataset_type}_uiuc",
                                        f"{sample['base_name']}.dat")
            shutil.copy2(sample["dat_path"], dst_dat_path)
    copy_files(train_samples, "train")
    copy_files(val_samples, "val")


def compute_r2(pred, true):
    pred = pred.detach()
    true = true.detach()
    if pred.shape[0] == 0:
        return 0.0
    ss_res = torch.sum((pred - true) ** 2, dim=1)
    true_mean = torch.mean(true, dim=1, keepdim=True)
    ss_tot = torch.sum((true - true_mean) ** 2, dim=1)
    ss_tot = torch.clamp(ss_tot, min=1e-8)
    r2_per_sample = 1 - (ss_res / ss_tot)
    r2_per_sample = torch.nan_to_num(r2_per_sample, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(r2_per_sample).item()

def compute_metrics(pred, true, criterion_mae, criterion_mse_per_coord, stats=None):
    actual_batch_size = pred.shape[0]
    if actual_batch_size == 0:
        if stats is not None:
            return None
        return {
            'avg_mse_per_coord': 0.0,
            'avg_mae_per_coord': 0.0,
            'avg_mse_per_sample': 0.0,
            'avg_mae_per_sample': 0.0,
            'r2': 0.0
        }
    num_coords = pred.shape[1] if len(pred.shape) > 1 else 1
    mse_per_coord = criterion_mse_per_coord(pred, true)
    mae_per_coord = criterion_mae(pred, true)
    mse_per_coord = torch.nan_to_num(mse_per_coord, nan=0.0, posinf=0.0, neginf=0.0)
    mae_per_coord = torch.nan_to_num(mae_per_coord, nan=0.0, posinf=0.0, neginf=0.0)
    mse_per_coord = mse_per_coord.detach()
    mae_per_coord = mae_per_coord.detach()
    batch_metrics = {
        'avg_mse_per_coord': mse_per_coord.mean().item(),
        'avg_mae_per_coord': mae_per_coord.mean().item(),
        'avg_mse_per_sample': mse_per_coord.sum(dim=1).mean().item(),
        'avg_mae_per_sample': mae_per_coord.sum(dim=1).mean().item(),
        'r2': compute_r2(pred, true)
    }
    if stats is not None:
        stats['total_mse_all_coords'] += mse_per_coord.sum().item()
        stats['total_mae_all_coords'] += mae_per_coord.sum().item()
        stats['total_mse_per_sample'] += mse_per_coord.sum(dim=1).sum().item()
        stats['total_mae_per_sample'] += mae_per_coord.sum(dim=1).sum().item()
        stats['total_r2'] += batch_metrics['r2'] * actual_batch_size
        stats['num_samples'] += actual_batch_size
        stats['total_coords'] += actual_batch_size * num_coords
        return None
    return batch_metrics
def train_val_model(
        split_data_root="./data/split_airfoil",
        epochs=1500,
        batch_size=32,
        lr=1e-4,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        attn_drop_rate=0.05,
        norm_layer=nn.LayerNorm,
        num_features=10,
        save_metrics=True,
        patience=50,
        monitor_metric='avg_mae_per_coord'
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        preload_device = device if torch.cuda.is_available() else torch.device('cpu')
    except:
        preload_device = torch.device('cpu')


    train_dataset = AirfoilDataset(
        dat_dir=os.path.join(split_data_root, "train_uiuc"),
        img_dir=os.path.join(split_data_root, "train_maps"),
        img_size=img_size,
        preload=True,
        device=preload_device
    )
    val_dataset = AirfoilDataset(
        dat_dir=os.path.join(split_data_root, "val_uiuc"),
        img_dir=os.path.join(split_data_root, "val_maps"),
        img_size=img_size,
        preload=True,
        device=preload_device
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False
    )

    num_y_coords = train_dataset[0][1].shape[0] if len(train_dataset) > 0 else 60
    model = AirfoilViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
        attn_drop_rate=attn_drop_rate,
        norm_layer=norm_layer,
        num_features=num_features,
        num_y_coords=num_y_coords
    ).to(device)

    criterion_mae = nn.L1Loss(reduction='none')
    criterion_mse_per_coord = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.95)

    metrics_history = {
        'epoch': [],
        'train_avg_mse_per_coord': [],
        'train_avg_mae_per_coord': [],
        'train_avg_mse_per_sample': [],
        'train_avg_mae_per_sample': [],
        'train_r2': [],
        'val_avg_mse_per_coord': [],
        'val_avg_mae_per_coord': [],
        'val_avg_mse_per_sample': [],
        'val_avg_mae_per_sample': [],
        'val_r2': []
    }
    best_metric = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_path = './model_10.pth'


    for epoch in range(epochs):

        model.train()
        train_stats = {
            'total_mse_all_coords': 0.0,
            'total_mae_all_coords': 0.0,
            'total_mse_per_sample': 0.0,
            'total_mae_per_sample': 0.0,
            'total_r2': 0.0,
            'num_samples': 0,
            'total_coords': 0
        }
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [训练]"):
            imgs, y_coords, _ = batch
            imgs, y_coords = imgs.to(device), y_coords.to(device)
            optimizer.zero_grad()
            _, pred_y = model(imgs)
            loss = criterion_mae(pred_y, y_coords).mean()
            loss.backward()
            optimizer.step()
            compute_metrics(pred_y, y_coords, criterion_mae, criterion_mse_per_coord, stats=train_stats)
        train_metrics = {
            'avg_mse_per_coord': train_stats['total_mse_all_coords'] / train_stats['total_coords'],
            'avg_mae_per_coord': train_stats['total_mae_all_coords'] / train_stats['total_coords'],
            'avg_mse_per_sample': train_stats['total_mse_per_sample'] / train_stats['num_samples'],
            'avg_mae_per_sample': train_stats['total_mae_per_sample'] / train_stats['num_samples'],
            'r2': train_stats['total_r2'] / train_stats['num_samples']
        }
        model.eval()
        val_stats = {
            'total_mse_all_coords': 0.0,
            'total_mae_all_coords': 0.0,
            'total_mse_per_sample': 0.0,
            'total_mae_per_sample': 0.0,
            'total_r2': 0.0,
            'num_samples': 0,
            'total_coords': 0
        }
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [验证]"):
                imgs, y_coords, _ = batch
                imgs, y_coords = imgs.to(device), y_coords.to(device)
                _, pred_y = model(imgs)
                compute_metrics(pred_y, y_coords, criterion_mae, criterion_mse_per_coord, stats=val_stats)
        val_metrics = {
            'avg_mse_per_coord': val_stats['total_mse_all_coords'] / val_stats['total_coords'],
            'avg_mae_per_coord': val_stats['total_mae_all_coords'] / val_stats['total_coords'],
            'avg_mse_per_sample': val_stats['total_mse_per_sample'] / val_stats['num_samples'],
            'avg_mae_per_sample': val_stats['total_mae_per_sample'] / val_stats['num_samples'],
            'r2': val_stats['total_r2'] / val_stats['num_samples']
        }
        current_metric = val_metrics[monitor_metric]
        best_val_r2 = 0.0
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_val_r2 = val_metrics['r2']
            patience_counter = 0
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'best_val_r2': val_metrics['r2'],
                'metrics': val_metrics
            }, best_model_path)
            print(f"Epoch {best_epoch} | {monitor_metric}：{best_metric:.8f} | R²：{best_val_r2:.8f}")
        else:
            patience_counter += 1
            print(
                f"{patience_counter}/{patience} | {monitor_metric}：{current_metric:.8f} | {best_metric:.8f} | R²：{val_metrics['r2']:.8f}")

        if patience_counter >= patience:

            break
        metrics_history['epoch'].append(epoch + 1)
        for key in ['avg_mse_per_coord', 'avg_mae_per_coord', 'avg_mse_per_sample', 'avg_mae_per_sample']:
            metrics_history[f'train_{key}'].append(train_metrics[key])
            metrics_history[f'val_{key}'].append(val_metrics[key])
        metrics_history['train_r2'].append(train_metrics['r2'])
        metrics_history['val_r2'].append(val_metrics['r2'])

        scheduler.step()
    if save_metrics:
        pd.DataFrame(metrics_history).to_csv('./training_validation_metrics_10.csv', index=False)
    return model, metrics_history

if __name__ == '__main__':

    # split_and_save_dataset(
    #     src_dat_dir="./data/uiuc_convert",
    #     src_img_dir="./data/airfoil_maps",
    #     dst_root="./data/split_airfoil",
    #     val_split=0.09,
    #     random_seed=0
    # )


    split_root="./data/split_airfoil"
    model, train_val_history = train_val_model(
        split_data_root=split_root,
        epochs=1500,
        batch_size=32,
        lr=1e-4,
        save_metrics=True,
        patience=500,
        num_features=10
    )

