from torch import nn
from torch.utils.data import Dataset
import torch
import os
import json
import numpy as np
import torch.nn.functional as F




class RobustAdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_cl = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cd = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred, target):
        pred_cl, pred_cd = pred[:, 0:1], pred[:, 1:2]
        true_cl, true_cd = target[:, 0:1], target[:, 1:2]

        loss_cl = F.l1_loss(pred_cl, true_cl)
        loss_cd = F.l1_loss(pred_cd, true_cd)

        sigma_cl = F.softplus(self.log_sigma_cl)
        sigma_cd = F.softplus(self.log_sigma_cd)

        loss = (loss_cl / sigma_cl + self.log_sigma_cl) + \
               (loss_cd / sigma_cd + self.log_sigma_cd)
        return loss
class RawMLPDataset(Dataset):
    def __init__(self, data_path, device='cuda'):
        super(RawMLPDataset, self).__init__()
        self.device = torch.device(device)
        try:
            npz_data = np.load(data_path)
            np_data = npz_data['data'].astype(np.float32)
            npz_data.close()

        except Exception as e:
            raise ValueError(f"error")
        self.raw_X = np_data[:, :12]
        self.raw_y = np_data[:, 12:14]
        self.cl_mean = np.mean(self.raw_y[:, 0])
        self.cl_sd = np.std(self.raw_y[:, 0])
        self.cd_mean = np.mean(self.raw_y[:, 1])
        self.cd_sd = np.std(self.raw_y[:, 1])
        self.X = torch.from_numpy(self.raw_X).float()
        self.y_cl = torch.from_numpy(self.raw_y[:, 0:1]).float()
        self.y_cd = torch.from_numpy(self.raw_y[:, 1:2]).float()
    def __len__(self):
        return len(self.raw_X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_cl[idx], self.y_cd[idx]
def huber_relative_error(pred, true, threshold=0.1, scale_factor=100.0):

    abs_err = torch.abs(pred - true).to(pred.device)
    abs_true = torch.abs(true).to(pred.device)

    small_value_mask = abs_true <= threshold

    rel_err = torch.zeros_like(true, dtype=torch.float32,device=pred.device)

    rel_err[small_value_mask] = abs_err[small_value_mask] * scale_factor / threshold

    rel_err[~small_value_mask] = (abs_err[~small_value_mask] / abs_true[~small_value_mask]) * scale_factor


    return rel_err
class MetricsCalculator:
    def __init__(self, save_path='metrics_logs', summary_filename='metrics.json'):
        self.save_path = save_path
        self.summary_filename = summary_filename
        self.reset()
        self.all_epoch_metrics = dict()
        os.makedirs(self.save_path, exist_ok=True)
    def reset(self):
        self.cl_mae_sum = 0.0
        self.cd_mae_sum = 0.0
        self.cl_mse_sum = 0.0
        self.cd_mse_sum = 0.0
        self.cl_rel_sum = 0.0
        self.cd_rel_sum = 0.0
        self.cl_true_sum = 0.0
        self.cl_true_sq_sum = 0.0
        self.cl_res_sum = 0.0
        self.cd_true_sum = 0.0
        self.cd_true_sq_sum = 0.0
        self.cd_res_sum = 0.0
        self.total_samples = 0
    def update(self, pred, true):
        cl_pred = pred[:, 0]
        cd_pred = pred[:, 1]
        cl_true = true[:, 0]
        cd_true = true[:, 1]
        cl_abs_err = torch.abs(cl_pred - cl_true)
        cd_abs_err = torch.abs(cd_pred - cd_true)
        self.cl_mae_sum += cl_abs_err.sum().item()
        self.cd_mae_sum += cd_abs_err.sum().item()
        self.cl_mse_sum += torch.square(cl_abs_err).sum().item()
        self.cd_mse_sum += torch.square(cd_abs_err).sum().item()
        cl_rel_err = huber_relative_error(cl_pred, cl_true, threshold=0.0977)
        cd_rel_err = huber_relative_error(cd_pred, cd_true, threshold=0.0078)
        self.cl_rel_sum += cl_rel_err.sum().item()
        self.cd_rel_sum += cd_rel_err.sum().item()
        self.cl_true_sum += cl_true.sum().item()
        self.cl_true_sq_sum += torch.square(cl_true).sum().item()
        self.cl_res_sum += torch.square(cl_abs_err).sum().item()
        self.cd_true_sum += cd_true.sum().item()
        self.cd_true_sq_sum += torch.square(cd_true).sum().item()
        self.cd_res_sum += torch.square(cd_abs_err).sum().item()
        self.total_samples += len(pred)
    def _calc_r2(self, res_sum, true_sum, true_sq_sum, n):
        if n == 0:
            return 0.0
        true_mean = true_sum / n
        tss = true_sq_sum - n * (true_mean ** 2)
        if tss < 1e-8:
            return 1.0
        r2 = 1 - (res_sum / tss)
        return max(0.0, min(1.0, r2))
    def compute(self, epoch):
        n = self.total_samples
        cl_mae = self.cl_mae_sum / n
        cd_mae = self.cd_mae_sum / n
        cl_mse = self.cl_mse_sum / n
        cd_mse = self.cd_mse_sum / n
        cl_rel = self.cl_rel_sum / n
        cd_rel = self.cd_rel_sum / n
        cl_r2 = self._calc_r2(self.cl_res_sum, self.cl_true_sum, self.cl_true_sq_sum, n)
        cd_r2 = self._calc_r2(self.cd_res_sum, self.cd_true_sum, self.cd_true_sq_sum, n)
        cl_rmse = np.sqrt(cl_mse)
        cd_rmse = np.sqrt(cd_mse)
        save_metrics = {
            'epoch': epoch,
            'cl_mae': round(cl_mae, 8),
            'cd_mae': round(cd_mae, 8),
            'cl_mse': round(cl_mse, 8),
            'cd_mse': round(cd_mse, 8),
            'cl_rel': round(cl_rel, 4),
            'cd_rel': round(cd_rel, 4),
            'cl_r2': round(cl_r2, 6),
            'cd_r2': round(cd_r2, 6),
        }
        print_metrics = {
            **save_metrics,
            'cl_rmse': round(cl_rmse, 8),
            'cd_rmse': round(cd_rmse, 8),
            'avg_mae': round((cl_mae + cd_mae) / 2, 8),
            'avg_mse': round((cl_mse + cd_mse) / 2, 8),
            'avg_rmse': round((cl_rmse + cd_rmse) / 2, 8),
            'avg_rel': round((cl_rel + cd_rel) / 2, 4),
            'avg_r2': round((cl_r2 + cd_r2) / 2, 6)
        }
        self.all_epoch_metrics[epoch] = save_metrics
        self.save_all_metrics()
        return print_metrics
    def save_all_metrics(self):
        path = os.path.join(self.save_path, self.summary_filename)
        sorted_list = sorted(self.all_epoch_metrics.values(), key=lambda x: x["epoch"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted_list, f, indent=2)
class EarlyStopping:
    def __init__(self, patience=20, save_path='best_model.pth'):
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_score = float('inf')
        self.early_stop = False
        self.best_train_metrics = None
        self.best_val_metrics = None
        self.best_epoch = None
    def __call__(self, val_avg_rel, train_metrics, val_metrics, model, epoch, optimizer):
        if val_avg_rel < self.best_score:
            self.best_score = val_avg_rel
            self.counter = 0
            self.best_train_metrics = train_metrics
            self.best_val_metrics = val_metrics
            self.best_epoch = epoch
            self._save_checkpoint(model, optimizer, val_avg_rel, epoch)

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    def _save_checkpoint(self, model, optimizer, val_avg_rel, epoch):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_avg_rel': val_avg_rel,
            'best_train_metrics': self.best_train_metrics,
            'best_val_metrics': self.best_val_metrics
        }
        save_dir = os.path.dirname(self.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(checkpoint, self.save_path)
    def print_final_best_metrics(self):

        print(f"Epoch {self.best_epoch} ")

        print("\n📊 train metrics:")
        t = self.best_train_metrics
        print(
            f"  CL - MAE: {t['cl_mae']:.8f}, MSE: {t['cl_mse']:.8f}, RMSE: {t['cl_rmse']:.8f}, Rel: {t['cl_rel']:.4f}%, R²: {t['cl_r2']:.6f}")
        print(
            f"  CD - MAE: {t['cd_mae']:.8f}, MSE: {t['cd_mse']:.8f}, RMSE: {t['cd_rmse']:.8f}, Rel: {t['cd_rel']:.4f}%, R²: {t['cd_r2']:.6f}")
        print(
            f"  mean - MAE: {t['avg_mae']:.8f}, MSE: {t['avg_mse']:.8f}, RMSE: {t['avg_rmse']:.8f}, Rel: {t['avg_rel']:.4f}%, R²: {t['avg_r2']:.6f}")
        print("\n📊 val metrics:")
        v = self.best_val_metrics
        print(
            f"  CL - MAE: {v['cl_mae']:.8f}, MSE: {v['cl_mse']:.8f}, RMSE: {v['cl_rmse']:.8f}, Rel: {v['cl_rel']:.4f}%, R²: {v['cl_r2']:.6f}")
        print(
            f"  CD - MAE: {v['cd_mae']:.8f}, MSE: {v['cd_mse']:.8f}, RMSE: {v['cd_rmse']:.8f}, Rel: {v['cd_rel']:.4f}%, R²: {v['cd_r2']:.6f}")
        print(
            f"  mean - MAE: {v['avg_mae']:.8f}, MSE: {v['avg_mse']:.8f}, RMSE: {v['avg_rmse']:.8f}, Rel: {v['avg_rel']:.4f}%, R²: {v['avg_r2']:.6f}")


