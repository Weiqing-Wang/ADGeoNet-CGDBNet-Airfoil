import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
import torch.nn.functional as F
from src.prediction.new.CG_DBNet import CGDBNet
from src.prediction.new.G_DBNet import GDBNet
from src.prediction.new.DBNet import DBNet
import os

from src.prediction.new.utils import RobustAdaptiveLoss, RawMLPDataset, MetricsCalculator, EarlyStopping

warnings.filterwarnings('ignore')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DATA_PATH = '../data/train_val_test_pre_data/train_data.npz'
VAL_DATA_PATH = '../data/train_val_test_pre_data/val_data.npz'
TEST_DATA_PATH = '../data/train_val_test_pre_data/test_data.npz'
BATCH_SIZE = 1024
EPOCHS = 500
LEARNING_RATE = 5e-4
PATIENCE = 500
MODEL_SAVE_PATH = f'best_model_one_new.pth'
METRICS_SAVE_PATH = f'best_model_one_new'
SEED = 42
MODEL_TYPE = "one"


def get_model():
    if MODEL_TYPE == "one":
        return CGDBNet().to(DEVICE)
    elif MODEL_TYPE == "two":
        return GDBNet().to(DEVICE)
    else:
        return DBNet().to(DEVICE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)



def train():
    set_seed(SEED)


    train_dataset = RawMLPDataset(data_path=TRAIN_DATA_PATH, device=DEVICE)
    val_dataset = RawMLPDataset(data_path=VAL_DATA_PATH, device=DEVICE)


    print(f"training - CL - mean: {train_dataset.cl_mean:.6f}, CL - std: {train_dataset.cl_sd:.6f}")
    print(f"training - CD - mean: {train_dataset.cd_mean:.6f}, CD - std: {train_dataset.cd_sd:.6f}")
    print(f"val - CL - mean: {val_dataset.cl_mean:.6f}, CL - std: {val_dataset.cl_sd:.6f}")
    print(f"val - CD - mean: {val_dataset.cd_mean:.6f}, CD - std: {val_dataset.cd_sd:.6f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    print(f"training: {len(train_dataset)} | val: {len(val_dataset)} ")

    model = get_model()
    loss_fn = RobustAdaptiveLoss().to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, step_size=25, gamma=0.80)


    train_metrics_calc = MetricsCalculator(save_path=METRICS_SAVE_PATH, summary_filename='train_metrics_new_pinn.json')
    val_metrics_calc = MetricsCalculator(save_path=METRICS_SAVE_PATH, summary_filename='val_metrics_new_pinn.json')
    test_metrics_calc = MetricsCalculator(save_path=METRICS_SAVE_PATH, summary_filename='test_metrics_new_pinn.json')

    early_stopping = EarlyStopping(patience=PATIENCE, save_path=MODEL_SAVE_PATH)


    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_metrics_calc.reset()
        train_loss_total = 0.0
        train_total_samples = 0

        for X, y_cl, y_cd in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]'):
            X = X.to(DEVICE, non_blocking=True)
            y_cl = y_cl.to(DEVICE, non_blocking=True)
            y_cd = y_cd.to(DEVICE, non_blocking=True)
            bs = X.shape[0]
            train_total_samples += bs

            optimizer.zero_grad()
            pred = model(X)
            target = torch.cat([y_cl, y_cd], dim=1)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * bs
            train_metrics_calc.update(pred, target)

        model.eval()
        val_metrics_calc.reset()
        val_loss_total = 0.0
        val_total_samples = 0

        with torch.no_grad():
            for X, y_cl, y_cd in tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]'):
                X = X.to(DEVICE, non_blocking=True)
                y_cl = y_cl.to(DEVICE, non_blocking=True)
                y_cd = y_cd.to(DEVICE, non_blocking=True)
                bs = X.shape[0]
                val_total_samples += bs

                pred = model(X)
                target = torch.cat([y_cl, y_cd], dim=1)
                loss = loss_fn(pred, target)

                val_loss_total += loss.item() * bs
                val_metrics_calc.update(pred, target)

        test_metrics_calc.reset()





        train_metrics = train_metrics_calc.compute(epoch)
        val_metrics = val_metrics_calc.compute(epoch)

        scheduler.step()

        train_loss_avg = train_loss_total / train_total_samples if train_total_samples > 0 else 0
        val_loss_avg = val_loss_total / val_total_samples if val_total_samples > 0 else 0

        print(f"\n===== Epoch {epoch}/{EPOCHS}  =====")

        print(f"📈 train loss: {train_loss_avg:.8f} | val loss: {val_loss_avg:.8f}")

        print("\n train metrics:")
        print(f"【CL】 MAE: {train_metrics['cl_mae']:.8f}, MSE: {train_metrics['cl_mse']:.8f}, RMSE: {train_metrics['cl_rmse']:.8f}, Rel: {train_metrics['cl_rel']:.4f}%, R²: {train_metrics['cl_r2']:.6f}")
        print(f"【CD】 MAE: {train_metrics['cd_mae']:.8f}, MSE: {train_metrics['cd_mse']:.8f}, RMSE: {train_metrics['cd_rmse']:.8f}, Rel: {train_metrics['cd_rel']:.4f}%, R²: {train_metrics['cd_r2']:.6f}")
        print(f"【平均】 MAE: {train_metrics['avg_mae']:.8f}, MSE: {train_metrics['avg_mse']:.8f}, RMSE: {train_metrics['avg_rmse']:.8f}, Rel: {train_metrics['avg_rel']:.4f}%, R²: {train_metrics['avg_r2']:.6f}")

        print("\n📊 val metrics:")
        print(f"【CL】 MAE: {val_metrics['cl_mae']:.8f}, MSE: {val_metrics['cl_mse']:.8f}, RMSE: {val_metrics['cl_rmse']:.8f}, Rel: {val_metrics['cl_rel']:.4f}%, R²: {val_metrics['cl_r2']:.6f}")
        print(f"【CD】 MAE: {val_metrics['cd_mae']:.8f}, MSE: {val_metrics['cd_mse']:.8f}, RMSE: {val_metrics['cd_rmse']:.8f}, Rel: {val_metrics['cd_rel']:.4f}%, R²: {val_metrics['cd_r2']:.6f}")
        print(f"【平均】 MAE: {val_metrics['avg_mae']:.8f}, MSE: {val_metrics['avg_mse']:.8f}, RMSE: {val_metrics['avg_rmse']:.8f}, Rel: {val_metrics['avg_rel']:.4f}%, R²: {val_metrics['avg_r2']:.6f}")




        if early_stopping(val_metrics['avg_rel'], train_metrics, val_metrics, model, epoch, optimizer):
            break

    early_stopping.print_final_best_metrics()



if __name__ == '__main__':
    train()