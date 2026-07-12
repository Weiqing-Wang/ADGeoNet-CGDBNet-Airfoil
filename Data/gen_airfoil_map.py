import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL.ImageOps as ops
from io import BytesIO


class AirfoilHeatmapGenerator:
    def __init__(self,
                 dat_dir="./uiuc_convert",
                 heatmap_dir="./airfoil_maps"):
        self.dat_dir = os.path.abspath(dat_dir)
        self.heatmap_dir = os.path.abspath(heatmap_dir)
        os.makedirs(self.heatmap_dir, exist_ok=True)
        self.single_save_dir = os.path.abspath(".")

        self.figsize = (6, 3)
        self.xlim = [-0.1, 1.1]
        self.ylim = [-0.2, 0.2]
        self.line_color = 'k'
        self.line_width = 3

    def read_airfoil_data(self, dat_path):
        x, y = [], []
        try:
            with open(dat_path, "rt") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
            return np.array(x), np.array(y) if x else (None, None)
        except Exception as e:
            return None, None

    def generate_raw_image(self, x, y):
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.plot(x, y, color=self.line_color, lw=self.line_width)
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        plt.axis('off')

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        return Image.open(buffer)

    def convert_to_gray_inverted(self, raw_img):
        gray_img = raw_img.convert('L')
        inverted_img = ops.invert(gray_img)
        return inverted_img

    def generate_heatmap(self, gray_img):
        img_array = np.array(gray_img)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img_array, cmap=None)
        plt.axis('off')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buffer.seek(0)

        heatmap_img = Image.open(buffer).convert('RGB')
        return heatmap_img

    def process_single_airfoil(self, file):
        dat_path = os.path.join(self.dat_dir, file)
        name = os.path.splitext(file)[0]
        save_path = os.path.join(self.heatmap_dir, f"{name}.png")

        x, y = self.read_airfoil_data(dat_path)
        if x is None or y is None:
            return False

        try:
            raw_img = self.generate_raw_image(x, y)
        except Exception as e:
            return False

        try:
            gray_img = self.convert_to_gray_inverted(raw_img)
            raw_img.close()
        except Exception as e:
            return False

        try:
            heatmap_img = self.generate_heatmap(gray_img)
            heatmap_img.save(save_path)
            gray_img.close()
            heatmap_img.close()
            return True
        except Exception as e:
            return False

    def process_single_airfoil_with_save_all(self, dat_file_path):

        file_name = os.path.basename(dat_file_path)
        name = os.path.splitext(file_name)[0]

        x, y = self.read_airfoil_data(dat_file_path)

        try:
            raw_img = self.generate_raw_image(x, y)
            raw_save_path = os.path.join(self.single_save_dir, f"{name}_raw.png")
            raw_img.save(raw_save_path)
        except Exception as e:
            return False

        try:
            gray_inverted_img = self.convert_to_gray_inverted(raw_img)
            gray_save_path = os.path.join(self.single_save_dir, f"{name}_gray_inverted.png")
            gray_inverted_img.save(gray_save_path)
            raw_img.close()
        except Exception as e:
            return False

        try:
            heatmap_img = self.generate_heatmap(gray_inverted_img)
            heatmap_save_path = os.path.join(self.single_save_dir, f"{name}_heatmap.png")
            heatmap_img.save(heatmap_save_path)
            gray_inverted_img.close()
            heatmap_img.close()
        except Exception as e:
            return False

        return True

    def batch_generate_heatmaps(self):
        dat_files = [f for f in os.listdir(self.dat_dir) if f.endswith('.dat')]

        total = len(dat_files)

        success = 0

        for i in range(total):
            file = dat_files[i]
            print(f"{i + 1} airfoil：{file}")
            if self.process_single_airfoil(file):
                success += 1



if __name__ == '__main__':
    generator = AirfoilHeatmapGenerator(
        dat_dir="./uiuc_convert",
        heatmap_dir="./airfoil_maps"
    )

    single_dat_file = "./uiuc_convert/rae100.dat"
    generator.process_single_airfoil_with_save_all(single_dat_file)

    # generator.batch_generate_heatmaps()