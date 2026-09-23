import os
import json
import numpy as np
from torch import nn
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import Dataset

class RawMLPDataset(Dataset):
    def __init__(self, data_path: str, preload: bool = True, device: str = "cpu"):
        """
        加载NPZ数据，支持预加载到GPU显存
        :param data_path: npz文件路径
        :param preload: 是否一次性把全部数据加载到目标设备(GPU)
        :param device: "cpu" / "cuda"
        """
        super().__init__()
        try:
            npz_data = np.load(data_path, mmap_mode="r")
            X_all = npz_data["X"].astype(np.float32)
            y_all = npz_data["y"].astype(np.float32)
        except Exception as e:
            raise ValueError(f"加载NPZ文件失败！错误：{e}\n请确认文件路径正确，NPZ包含 X、y 键")

        assert X_all.shape[0] == y_all.shape[0], "X与标签样本数量不一致"

        # 先转成torch张量
        self.X = torch.from_numpy(X_all)
        self.y_cl = self.X.new_zeros(X_all.shape[0])
        self.y_cd = self.X.new_zeros(X_all.shape[0])
        # 从y取CL、CD
        self.y_cl = torch.from_numpy(y_all[:, 0])
        self.y_cd = torch.from_numpy(y_all[:, 1])

        # 预加载：把全部数据一次性搬到目标设备
        if preload:
            self.X = self.X.to(device)
            self.y_cl = self.y_cl.to(device)
            self.y_cd = self.y_cd.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y_cl = self.y_cl[idx:idx+1]
        y_cd = self.y_cd[idx:idx+1]
        return x, y_cl, y_cd



def robust_relative_percent_metric(pred, true, threshold=0.1, scale_factor=100.0):
    """
    真值阈值保护的等效相对误差评价指标（输出为百分比，仅用于评估，不做损失）
    当 |true| ≤ threshold：使用 threshold 作为分母避免除零爆炸
    当 |true| > threshold：使用标准相对误差
    pred, true: torch.Tensor, shape [N]
    return: 等效相对误差(百分比)，shape [N]
    """
    abs_err = torch.abs(pred - true).to(pred.device)
    abs_true = torch.abs(true).to(pred.device)

    small_value_mask = abs_true <= threshold
    rel_err = torch.zeros_like(true, dtype=torch.float32, device=pred.device)

    # 小真值区域：等效相对误差
    rel_err[small_value_mask] = abs_err[small_value_mask] * scale_factor / threshold
    # 大真值区域：标准百分比相对误差
    rel_err[~small_value_mask] = (abs_err[~small_value_mask] / abs_true[~small_value_mask]) * scale_factor
    return rel_err

class CipollaLaplaceLoss(nn.Module):
    """
    Kendall & Cipolla 2017 Laplace 负对数似然损失
    L = E[ |y‑ŷ| / σ + logσ ]
    参数化：s = logσ，σ = exp(s)
    """
    def __init__(self):
        super().__init__()
        self.log_sigma_cl = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_cd = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred, target):
        pred_cl, pred_cd = pred[:, 0:1], pred[:, 1:2]
        true_cl, true_cd = target[:, 0:1], target[:, 1:2]

        err_cl = torch.abs(pred_cl - true_cl)
        err_cd = torch.abs(pred_cd - true_cd)

        sigma_cl = torch.exp(self.log_sigma_cl)
        sigma_cd = torch.exp(self.log_sigma_cd)

        nll_cl = err_cl / sigma_cl + self.log_sigma_cl
        nll_cd = err_cd / sigma_cd + self.log_sigma_cd

        loss = torch.mean(nll_cl + nll_cd)
        return loss

    def get_sigma_values(self):
        sigma_cl = torch.exp(self.log_sigma_cl).item()
        sigma_cd = torch.exp(self.log_sigma_cd).item()
        return sigma_cl, sigma_cd

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

        cl_rel_err = robust_relative_percent_metric(cl_pred, cl_true, threshold=0.0976)
        cd_rel_err = robust_relative_percent_metric(cd_pred, cd_true, threshold=0.0078)
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
        r2 = 1.0 - (res_sum / tss)
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

        # json保存原始浮点，不做round
        save_metrics = {
            'epoch': epoch,
            'cl_mae': cl_mae,
            'cd_mae': cd_mae,
            'cl_mse': cl_mse,
            'cd_mse': cd_mse,
            'cl_rel': cl_rel,
            'cd_rel': cd_rel,
            'cl_r2': cl_r2,
            'cd_r2': cd_r2,
            'cl_rmse': cl_rmse,
            'cd_rmse': cd_rmse
        }
        # 打印做round
        print_metrics = {
            "epoch": epoch,
            "cl_mae": round(cl_mae, 8),
            "cd_mae": round(cd_mae, 8),
            "cl_mse": round(cl_mse, 8),
            "cd_mse": round(cd_mse, 8),
            "cl_rel": round(cl_rel,4),   # 转为百分比用于打印
            "cd_rel": round(cd_rel,4),
            "cl_r2": round(cl_r2,6),
            "cd_r2": round(cd_r2,6),
            "cl_rmse": round(cl_rmse,8),
            "cd_rmse": round(cd_rmse,8),
            "avg_mae": round((cl_mae+cd_mae)/2,8),
            "avg_mse": round((cl_mse+cd_mse)/2,8),
            "avg_rmse": round((cl_rmse+cd_rmse)/2,8),
            "avg_rel": round((cl_rel+cd_rel)/2,4),
            "avg_r2": round((cl_r2+cd_r2)/2,6)
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
        self.best_sigma_cl = None
        self.best_sigma_cd = None

    def __call__(self, val_avg_rel, train_metrics, val_metrics, model, loss_fn, optimizer, epoch, sigma_cl, sigma_cd):
        if val_avg_rel < self.best_score:
            self.best_score = val_avg_rel
            self.counter = 0
            self.best_train_metrics = train_metrics
            self.best_val_metrics = val_metrics
            self.best_epoch = epoch
            self.best_sigma_cl = sigma_cl
            self.best_sigma_cd = sigma_cd
            self._save_checkpoint(model, loss_fn, optimizer, val_avg_rel, epoch)
            print(f"✅ Epoch {epoch} | 最佳验证相对误差: {self.best_score:.4f} % | 模型已保存")
        else:
            self.counter += 1
            print(f"⚠️  Epoch {epoch} | 早停计数器 {self.counter}/{self.patience} | 最佳误差: {self.best_score:.4f} %")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def _save_checkpoint(self, model, loss_fn, optimizer, val_avg_rel, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_fn_state_dict': loss_fn.state_dict(),   # 保存Cipolla损失可学习参数
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_avg_rel': val_avg_rel,
            'best_sigma_cl': self.best_sigma_cl,
            'best_sigma_cd': self.best_sigma_cd,
            'best_train_metrics': self.best_train_metrics,
            'best_val_metrics': self.best_val_metrics
        }
        save_dir = os.path.dirname(self.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(checkpoint, self.save_path)

    def print_final_best_metrics(self):
        print("\n" + "=" * 80)
        print(f"🏆 最佳模型指标（Epoch {self.best_epoch} | 原始尺度）")
        print(f"sigma_cl={self.best_sigma_cl:.6f}, sigma_cd={self.best_sigma_cd:.6f}")
        print("=" * 80)
        print("\n📊 训练集指标:")
        t = self.best_train_metrics
        print(
            f"  CL - MAE: {t['cl_mae']:.8f}, MSE: {t['cl_mse']:.8f}, RMSE: {t['cl_rmse']:.8f}, Rel: {t['cl_rel']:.4f}%, R²: {t['cl_r2']:.6f}")
        print(
            f"  CD - MAE: {t['cd_mae']:.8f}, MSE: {t['cd_mse']:.8f}, RMSE: {t['cd_rmse']:.8f}, Rel: {t['cd_rel']:.4f}%, R²: {t['cd_r2']:.6f}")
        print(
            f"  平均 - MAE: {t['avg_mae']:.8f}, MSE: {t['avg_mse']:.8f}, RMSE: {t['avg_rmse']:.8f}, Rel: {t['avg_rel']:.4f}%, R²: {t['avg_r2']:.6f}")
        print("\n📊 验证集指标:")
        v = self.best_val_metrics
        print(
            f"  CL - MAE: {v['cl_mae']:.8f}, MSE: {v['cl_mse']:.8f}, RMSE: {v['cl_rmse']:.8f}, Rel: {v['cl_rel']:.4f}%, R²: {v['cl_r2']:.6f}")
        print(
            f"  CD - MAE: {v['cd_mae']:.8f}, MSE: {v['cd_mse']:.8f}, RMSE: {v['cd_rmse']:.8f}, Rel: {v['cd_rel']:.4f}%, R²: {v['cd_r2']:.6f}")
        print(
            f"  平均 - MAE: {v['avg_mae']:.8f}, MSE: {v['avg_mse']:.8f}, RMSE: {v['avg_rmse']:.8f}, Rel: {v['avg_rel']:.4f}%, R²: {v['avg_r2']:.6f}")
        print("=" * 80)
