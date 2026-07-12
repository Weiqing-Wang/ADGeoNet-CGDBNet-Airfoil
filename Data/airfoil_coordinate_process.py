"""
翼型坐标加密处理工具（核心：坐标顺序与原始完全一致 + 前缘密集）
关键修正：
1. 生成采样点后强制对齐原始坐标顺序（上表面TE→LE，下表面LE→TE）
2. 保留所有加密逻辑、可视化逻辑、参数配置不变
3. 仅修正坐标顺序，不改变加密密度、原始前缘/后缘坐标
新增功能：单个翼型处理时自动保存可视化图片到本地（PNG格式）
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from tqdm import tqdm


def read_airfoil_data(file_path):
    """读取原始数据，仅过滤空行和注释"""
    try:
        with open(file_path, 'r') as f:
            lines = []
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    lines.append(stripped)
        data = np.array([list(map(float, line.split())) for line in lines], dtype=np.float64)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"❌ 读取文件 {os.path.basename(file_path)} 失败：{str(e)}")
        return None, None


def split_upper_lower(x, y):
    """分离上下表面+单表面去重+强制方向（保留原始逻辑）"""
    le_idx = np.argmin(x)  # 前缘：x最小的点
    le_x = x[le_idx]
    le_y = y[le_idx]
    te_x = np.max(x)  # 后缘：x最大的点

    upper_x = x[:le_idx + 1].copy()
    upper_y = y[:le_idx + 1].copy()
    lower_x = x[le_idx:].copy()
    lower_y = y[le_idx:].copy()

    # 去重
    upper_unique_idx = np.unique(upper_x, return_index=True)[1]
    upper_unique_idx.sort()
    upper_x = upper_x[upper_unique_idx]
    upper_y = upper_y[upper_unique_idx]

    lower_unique_idx = np.unique(lower_x, return_index=True)[1]
    lower_unique_idx.sort()
    lower_x = lower_x[lower_unique_idx]
    lower_y = lower_y[lower_unique_idx]

    # 强制方向：上表面后缘→前缘（x递减），下表面前缘→后缘（x递增）
    if not np.all(np.diff(upper_x) <= 1e-6):
        upper_sort_idx = np.argsort(upper_x)[::-1]
        upper_x = upper_x[upper_sort_idx]
        upper_y = upper_y[upper_sort_idx]
    if not np.all(np.diff(lower_x) >= -1e-6):
        lower_sort_idx = np.argsort(lower_x)
        lower_x = lower_x[lower_sort_idx]
        lower_y = lower_y[lower_sort_idx]

    return upper_x, upper_y, lower_x, lower_y, le_x, le_y, te_x


def generate_le_dense_template(n_points, dense_factor):
    """生成前缘密集的统一疏密模板"""
    t_uniform = np.linspace(0, 1, n_points)
    t_dense = t_uniform ** dense_factor  # 核心：t值越大（前缘），点越密
    return t_dense


def generate_leading_edge_dense_points(x_orig, y_orig, n_points, dense_factor, is_upper=True):
    """
    生成前缘密集的采样点（核心修正上表面映射逻辑）
    :param is_upper: 是否为上表面（用于调整映射方向）
    """
    # 1. 生成前缘密集的统一模板
    t_dense = generate_le_dense_template(n_points, dense_factor)

    # 2. 保留原始表面的起止点
    x_start, x_end = x_orig[0], x_orig[-1]  # 上表面：后缘→前缘；下表面：前缘→后缘
    y_start, y_end = y_orig[0], y_orig[-1]

    # 3. 核心修正：上表面反转模板映射方向，确保前缘（x_end）侧更密
    if is_upper:
        x_dense = x_end + (x_start - x_end) * t_dense  # 上表面：前缘(x_end)→后缘(x_start)的密集映射
    else:
        x_dense = x_start + (x_end - x_start) * t_dense  # 下表面保持原逻辑

    # 4. 插值y值
    f_y = interp1d(x_orig, y_orig, kind='linear', fill_value="extrapolate")
    y_dense = f_y(x_dense)

    # 5. 强制前缘点与原始一致
    le_idx_orig = np.argmin(x_orig)
    le_idx_dense = np.argmin(x_dense)
    x_dense[le_idx_dense] = x_orig[le_idx_orig]
    y_dense[le_idx_dense] = y_orig[le_idx_orig]

    # ========== 新增：强制坐标顺序与原始一致 ==========
    if is_upper:
        # 上表面：强制从后缘（x最大值）→前缘（x最小值）（与原始顺序一致）
        sort_idx = np.argsort(x_dense)[::-1]  # x从大到小排序
        x_dense = x_dense[sort_idx]
        y_dense = y_dense[sort_idx]
    else:
        # 下表面：强制从前缘（x最小值）→后缘（x最大值）（与原始顺序一致）
        sort_idx = np.argsort(x_dense)  # x从小到大排序
        x_dense = x_dense[sort_idx]
        y_dense = y_dense[sort_idx]

    return x_dense, y_dense


def process_single_airfoil(input_path, output_dir="./uiuc_convert",
                           n_points_per_surface=30, dense_factor=2.0, plot=True,
                           save_figure=True, figure_save_dir="./airfoil_plots"):
    """
    处理单个翼型（移除第二个子图+修正上表面加密+保证坐标顺序）
    :param save_figure: 是否保存可视化图片
    :param figure_save_dir: 图片保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    # 创建图片保存目录
    if save_figure:
        os.makedirs(figure_save_dir, exist_ok=True)

    file_name = os.path.basename(input_path)

    x_orig, y_orig = read_airfoil_data(input_path)
    if x_orig is None or y_orig is None:
        return False
    print(f"📌 读取成功：{file_name}，原始总点数：{len(x_orig)}")

    try:
        upper_x, upper_y, lower_x, lower_y, le_x, le_y, te_x = split_upper_lower(x_orig, y_orig)
        print(f"📌 分离成功：前缘({le_x:.6f}, {le_y:.6f})，后缘x={te_x:.6f}")
        print(f"📌 分离后：上表面{len(upper_x)}点，下表面{len(lower_y)}点")
    except Exception as e:
        print(f"❌ 分离失败：{file_name} - {str(e)}")
        return False

    # 生成采样点（上表面指定is_upper=True，修正加密方向）
    upper_x_dense, upper_y_dense = generate_leading_edge_dense_points(
        upper_x, upper_y, n_points_per_surface, dense_factor, is_upper=True
    )
    lower_x_dense, lower_y_dense = generate_leading_edge_dense_points(
        lower_x, lower_y, n_points_per_surface, dense_factor, is_upper=False
    )

    # 验证间距（上表面：后缘→前缘，间距递减；下表面：前缘→后缘，间距递增）
    upper_spacing = np.diff(upper_x_dense)
    lower_spacing = np.diff(lower_x_dense)
    print(f"✅ 上表面X顺序：{upper_x_dense[0]:.6f} → {upper_x_dense[-1]:.6f}（后缘→前缘，与原始一致）")
    print(f"✅ 下表面X顺序：{lower_x_dense[0]:.6f} → {lower_x_dense[-1]:.6f}（前缘→后缘，与原始一致）")
    print(f"✅ 上表面X间距（后缘→前缘）：{np.abs(upper_spacing[0]):.6f} → {np.abs(upper_spacing[-1]):.6f}（递减，前缘更密）")
    print(f"✅ 下表面X间距（前缘→后缘）：{lower_spacing[0]:.6f} → {lower_spacing[-1]:.6f}（递增，前缘更密）")

    # 拼接上下表面
    final_x = np.concatenate([upper_x_dense, lower_x_dense])
    final_y = np.concatenate([upper_y_dense, lower_y_dense])

    # 强制上下表面前缘点一致
    le_pos = len(upper_x_dense) - 1
    final_x[le_pos] = le_x
    final_x[le_pos + 1] = le_x
    final_y[le_pos] = le_y
    final_y[le_pos + 1] = le_y

    # 容错：强制60点
    if len(final_x) != n_points_per_surface * 2:
        print(f"⚠️ 点数异常：预期{n_points_per_surface * 2}点，实际{len(final_x)}点")
        if len(final_x) > 60:
            final_x = final_x[:60]
            final_y = final_y[:60]
        else:
            pad_len = 60 - len(final_x)
            final_x = np.pad(final_x, (0, pad_len), mode='edge')
            final_y = np.pad(final_y, (0, pad_len), mode='edge')

    # 保存文件
    output_path = os.path.join(output_dir, file_name)
    try:
        with open(output_path, 'w') as f:
            for x, y in zip(final_x, final_y):
                f.write(f"{x:.6f} {y:.6f}\n")
        print(f"📁 保存成功：{os.path.abspath(output_path)}，最终总点数：{len(final_x)}")
    except Exception as e:
        print(f"❌ 保存失败：{file_name} - {str(e)}")
        return False

    if plot:
        # 仅保留翼型形状+采样点分布的可视化（移除第二个子图）
        fig, ax = plt.subplots(1, 1, figsize=[8, 5])

        # 绘制原始翼型和处理后的采样点
        ax.plot(x_orig, y_orig, 'k-', c="green", lw=1, label='Original Airfoil')
        # ax.scatter(final_x, final_y, c='red', s=30, alpha=0.8, label='Sampled Points (60)')

        # 标注前缘、后缘
        # ax.scatter(le_x, le_y, color='green', s=80, zorder=5, label='Leading Edge (LE)')
        # ax.scatter(te_x, y_orig[np.argmax(x_orig)], color='green', s=80, zorder=5, label='Trailing Edge (TE)')

        # 标注上下表面采样点
        ax.scatter(upper_x_dense, upper_y_dense, c='red', s=40, alpha=0.6, label='Upper Surface (30 points)')
        ax.scatter(lower_x_dense, lower_y_dense, c='blue', s=40, alpha=0.6, label='Lower Surface (30 points)')

        # 图表配置
        ax.set_xlabel('x/c',fontsize=20)
        ax.set_ylabel('y',fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        # ax.set_title(f"{file_name}")
        ax.legend(loc='best',fontsize=20,frameon=False)
        # ax.grid(alpha=0.3)
        ax.axis('equal')
        # ax.set_ylim(-0.2, 0.2)
        plt.tight_layout()

        # 保存图片到本地
        if save_figure:
            # 构建图片文件名（替换.dat后缀为.png）
            fig_file_name = os.path.splitext(file_name)[0] + '.png'
            fig_save_path = os.path.join(figure_save_dir, fig_file_name)
            # 保存图片（设置高分辨率）
            plt.savefig(fig_save_path, dpi=600, bbox_inches='tight')
            print(f"🖼️ 图片保存成功：{os.path.abspath(fig_save_path)}")

        plt.show()

    print(f"✅ 处理完成：{file_name}，最终总点数：{len(final_x)}\n")
    return True


def batch_process_airfoils(input_dir, output_dir="./uiuc_convert",
                           n_points_per_surface=30, dense_factor=2.0, plot=False):
    """批量处理翼型"""
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录 {input_dir} 不存在")
        return

    dat_files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]
    if not dat_files:
        print(f"⚠️ 目录 {input_dir} 中未找到.dat文件")
        return

    total = len(dat_files)
    success_count = 0
    fail_count = 0
    fail_files = []

    print("=" * 60)
    print(f"批量处理配置（前缘密集）：")
    print(f"输入目录：{os.path.abspath(input_dir)}")
    print(f"输出目录：{os.path.abspath(output_dir)}")
    print(f"文件总数：{total}，每个表面采样数：{n_points_per_surface}，加密强度：{dense_factor}")
    print(f"输出规格：60点（上下各30，前缘密集，保留原始LE/TE坐标，顺序与原始一致）")
    print("=" * 60 + "\n")

    for file_name in tqdm(dat_files, desc="处理进度"):
        file_path = os.path.join(input_dir, file_name)
        try:
            result = process_single_airfoil(
                input_path=file_path,
                output_dir=output_dir,
                n_points_per_surface=n_points_per_surface,
                dense_factor=dense_factor,
                plot=plot,
                save_figure=False  # 批量处理时默认不保存图片
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
                fail_files.append(file_name)
        except Exception as e:
            print(f"❌ 处理 {file_name} 意外错误：{str(e)}")
            fail_count += 1
            fail_files.append(file_name)

    print("\n" + "=" * 60)
    print(f"批量处理统计：")
    print(f"总文件数：{total}")
    print(f"成功处理：{success_count} 个")
    print(f"处理失败：{fail_count} 个")
    if fail_files:
        print(f"失败文件：{', '.join(fail_files)}")
    print(f"输出特征：上/下表面前缘均密集，疏密比例一致，保留原始LE/TE坐标，坐标顺序与原始完全一致")
    print("=" * 60)
    print(f"成功文件保存至：{os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # 配置参数
    SINGLE_FILE_PATH = "./uiuc/rae100.dat"  # 单个翼型文件路径
    BATCH_INPUT_DIR = "./uiuc"  # 批量输入目录
    BATCH_OUTPUT_DIR = "./uiuc_convert"  # 批量输出目录
    FIGURE_SAVE_DIR = "."  # 图片保存目录
    N_POINTS_PER_SURFACE = 30  # 每个表面30点（总60点）
    DENSE_FACTOR = 1.5  # 加密强度（1.5-3.0为宜）

    # 处理单个翼型（验证上表面前缘加密效果+坐标顺序，同时保存图片）
    print("===== 处理单个翼型（验证前缘密集+坐标顺序） =====")
    process_single_airfoil(
        input_path=SINGLE_FILE_PATH,
        output_dir=BATCH_OUTPUT_DIR,
        n_points_per_surface=N_POINTS_PER_SURFACE,
        dense_factor=DENSE_FACTOR,
        plot=True,
        save_figure=True,
        figure_save_dir=FIGURE_SAVE_DIR
    )
    # 备注：2025年12月10日，翼型图像生成采样的是本文件方法

    # 批量处理所有翼型（关闭可视化）
    # print("\n===== 批量处理翼型 =====")
    # batch_process_airfoils(
    #     input_dir=BATCH_INPUT_DIR,
    #     output_dir=BATCH_OUTPUT_DIR,
    #     n_points_per_surface=N_POINTS_PER_SURFACE,
    #     dense_factor=DENSE_FACTOR,
    #     plot=False
    # )