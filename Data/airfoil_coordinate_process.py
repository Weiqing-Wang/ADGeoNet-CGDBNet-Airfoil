

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from tqdm import tqdm


def read_airfoil_data(file_path):
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
        return None


def split_upper_lower(x, y):
    le_idx = np.argmin(x)
    le_x = x[le_idx]
    le_y = y[le_idx]
    te_x = np.max(x)

    upper_x = x[:le_idx + 1].copy()
    upper_y = y[:le_idx + 1].copy()
    lower_x = x[le_idx:].copy()
    lower_y = y[le_idx:].copy()


    upper_unique_idx = np.unique(upper_x, return_index=True)[1]
    upper_unique_idx.sort()
    upper_x = upper_x[upper_unique_idx]
    upper_y = upper_y[upper_unique_idx]

    lower_unique_idx = np.unique(lower_x, return_index=True)[1]
    lower_unique_idx.sort()
    lower_x = lower_x[lower_unique_idx]
    lower_y = lower_y[lower_unique_idx]

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
    t_uniform = np.linspace(0, 1, n_points)
    t_dense = t_uniform ** dense_factor
    return t_dense


def generate_leading_edge_dense_points(x_orig, y_orig, n_points, dense_factor, is_upper=True):

    t_dense = generate_le_dense_template(n_points, dense_factor)


    x_start, x_end = x_orig[0], x_orig[-1]
    y_start, y_end = y_orig[0], y_orig[-1]

    if is_upper:
        x_dense = x_end + (x_start - x_end) * t_dense
    else:
        x_dense = x_start + (x_end - x_start) * t_dense


    f_y = interp1d(x_orig, y_orig, kind='linear', fill_value="extrapolate")
    y_dense = f_y(x_dense)


    le_idx_orig = np.argmin(x_orig)
    le_idx_dense = np.argmin(x_dense)
    x_dense[le_idx_dense] = x_orig[le_idx_orig]
    y_dense[le_idx_dense] = y_orig[le_idx_orig]

    if is_upper:
        sort_idx = np.argsort(x_dense)[::-1]
        x_dense = x_dense[sort_idx]
        y_dense = y_dense[sort_idx]
    else:
        sort_idx = np.argsort(x_dense)
        x_dense = x_dense[sort_idx]
        y_dense = y_dense[sort_idx]

    return x_dense, y_dense


def process_single_airfoil(input_path, output_dir="./uiuc_convert",
                           n_points_per_surface=30, dense_factor=2.0, plot=True,
                           save_figure=True, figure_save_dir="./airfoil_plots"):

    os.makedirs(output_dir, exist_ok=True)
    if save_figure:
        os.makedirs(figure_save_dir, exist_ok=True)

    file_name = os.path.basename(input_path)

    x_orig, y_orig = read_airfoil_data(input_path)
    if x_orig is None or y_orig is None:
        return False


    try:
        upper_x, upper_y, lower_x, lower_y, le_x, le_y, te_x = split_upper_lower(x_orig, y_orig)

    except Exception as e:
        return False

    upper_x_dense, upper_y_dense = generate_leading_edge_dense_points(
        upper_x, upper_y, n_points_per_surface, dense_factor, is_upper=True
    )
    lower_x_dense, lower_y_dense = generate_leading_edge_dense_points(
        lower_x, lower_y, n_points_per_surface, dense_factor, is_upper=False
    )


    final_x = np.concatenate([upper_x_dense, lower_x_dense])
    final_y = np.concatenate([upper_y_dense, lower_y_dense])

    le_pos = len(upper_x_dense) - 1
    final_x[le_pos] = le_x
    final_x[le_pos + 1] = le_x
    final_y[le_pos] = le_y
    final_y[le_pos + 1] = le_y

    if len(final_x) != n_points_per_surface * 2:
        if len(final_x) > 60:
            final_x = final_x[:60]
            final_y = final_y[:60]
        else:
            pad_len = 60 - len(final_x)
            final_x = np.pad(final_x, (0, pad_len), mode='edge')
            final_y = np.pad(final_y, (0, pad_len), mode='edge')

    output_path = os.path.join(output_dir, file_name)
    try:
        with open(output_path, 'w') as f:
            for x, y in zip(final_x, final_y):
                f.write(f"{x:.6f} {y:.6f}\n")
        print(f"save：{os.path.abspath(output_path)}，total：{len(final_x)}")
    except Exception as e:
        print(f"fail：{file_name} - {str(e)}")
        return False

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=[8, 5])

        ax.plot(x_orig, y_orig, 'k-', c="green", lw=1, label='Original Airfoil')


        ax.scatter(upper_x_dense, upper_y_dense, c='red', s=40, alpha=0.6, label='Upper Surface (30 points)')
        ax.scatter(lower_x_dense, lower_y_dense, c='blue', s=40, alpha=0.6, label='Lower Surface (30 points)')

        ax.set_xlabel('x/c',fontsize=20)
        ax.set_ylabel('y',fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(loc='best',fontsize=20,frameon=False)
        ax.axis('equal')
        plt.tight_layout()

        if save_figure:
            fig_file_name = os.path.splitext(file_name)[0] + '.png'
            fig_save_path = os.path.join(figure_save_dir, fig_file_name)
            plt.savefig(fig_save_path, dpi=600, bbox_inches='tight')
            print(f"success：{os.path.abspath(fig_save_path)}")

        plt.show()

    return True


def batch_process_airfoils(input_dir, output_dir="./uiuc_convert",
                           n_points_per_surface=30, dense_factor=2.0, plot=False):


    dat_files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]


    total = len(dat_files)
    success_count = 0
    fail_count = 0
    fail_files = []



    for file_name in tqdm(dat_files, ):
        file_path = os.path.join(input_dir, file_name)
        try:
            result = process_single_airfoil(
                input_path=file_path,
                output_dir=output_dir,
                n_points_per_surface=n_points_per_surface,
                dense_factor=dense_factor,
                plot=plot,
                save_figure=False
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
                fail_files.append(file_name)
        except Exception as e:
            fail_count += 1
            fail_files.append(file_name)





if __name__ == "__main__":
    SINGLE_FILE_PATH = "./uiuc/rae100.dat"
    BATCH_INPUT_DIR = "./uiuc"
    BATCH_OUTPUT_DIR = "./uiuc_convert"
    FIGURE_SAVE_DIR = "."
    N_POINTS_PER_SURFACE = 30
    DENSE_FACTOR = 1.5


    process_single_airfoil(
        input_path=SINGLE_FILE_PATH,
        output_dir=BATCH_OUTPUT_DIR,
        n_points_per_surface=N_POINTS_PER_SURFACE,
        dense_factor=DENSE_FACTOR,
        plot=True,
        save_figure=True,
        figure_save_dir=FIGURE_SAVE_DIR
    )
