import numpy as np
import math
import cv2
from pyproj import Proj, transform,Transformer
from osgeo import osr, gdal
from exif2pos import Get_Image_angle
import os
import warnings


class ImageCorrection:
    def __init__(self, f, s, xfix=0, yfix=0):
        self.f = f  # 焦距(mm)
        self.s = s  # 像素大小(mm/pixel)
        self.xfix = xfix
        self.yfix = yfix

    def set_parameters(self, B, L, H, omega_rad, phi_rad, kappa_rad):
        self.B = B
        self.L = L
        self.H = H
        self.omega_rad = omega_rad
        self.phi_rad = phi_rad
        self.kappa_rad = kappa_rad
        self.d = self.s * self.H / self.f  # GSD地面分辨率（m/pixel）

        self.a1 = math.cos(self.phi_rad) * math.cos(self.kappa_rad) - math.sin(self.phi_rad) * math.sin(self.omega_rad) * math.sin(self.kappa_rad)
        self.a2 = -math.cos(self.phi_rad) * math.sin(self.kappa_rad) - math.sin(self.phi_rad) * math.sin(self.omega_rad) * math.cos(self.kappa_rad)
        self.a3 = -math.sin(self.phi_rad) * math.cos(self.omega_rad)
        self.b1 = math.cos(self.omega_rad) * math.sin(self.kappa_rad)
        self.b2 = math.cos(self.omega_rad) * math.cos(self.kappa_rad)
        self.b3 = -math.sin(self.omega_rad)
        self.c1 = math.sin(self.phi_rad) * math.cos(self.kappa_rad) + math.cos(self.phi_rad) * math.sin(self.omega_rad) * math.sin(self.kappa_rad)
        self.c2 = -math.sin(self.phi_rad) * math.sin(self.kappa_rad) + math.cos(self.phi_rad) * math.sin(self.omega_rad) * math.cos(self.kappa_rad)
        self.c3 = math.cos(self.phi_rad) * math.cos(self.omega_rad)

    def pos(self, x, y):
        """
        像平面坐标转换空间坐标
        """
        x = x - self.x0
        y = y - self.y0
        denominator = self.c1 * x + self.c2 * y - self.c3 * self.f  # 避免重复计算
        X = -self.H * (self.a1 * x + self.a2 * y - self.a3 * self.f) / denominator + self.Xs
        Y = -self.H * (self.b1 * x + self.b2 * y - self.b3 * self.f) / denominator + self.Ys
        return X, Y

    def pos_t(self, i, j):
        """
        求校正影像行列坐标对应原始影像的行列坐标
        """
        X = self.Xmin + j * self.d
        Y = self.Ymax - i * self.d

        den = self.a3 * (X - self.Xs) + self.b3 * (Y - self.Ys) - self.c3 * self.H
        x = -self.f * (self.a1 * (X - self.Xs) + self.b1 * (Y - self.Ys) - self.c1 * self.H) / den
        y = -self.f * (self.a2 * (X - self.Xs) + self.b2 * (Y - self.Ys) - self.c2 * self.H) / den
        x = x + self.x0
        y = y + self.y0

        x = x / self.s + self.src_w / 2
        y = self.src_h / 2 - y / self.s

        return y, x

    def nearest(self, src_img, dst_shape):
        """
        最近邻
        """
        dst_img = np.zeros((dst_shape[0], dst_shape[1], 3), np.uint8)
        dst_h, dst_w = dst_shape

        src_x, src_y = self.pos_t(np.arange(dst_h)[:, None], np.arange(dst_w))
        src_x = src_x.astype(int)
        src_y = src_y.astype(int)

        valid_mask = (src_x >= 0) & (src_y >= 0) & (src_x < self.src_h) & (src_y < self.src_w)

        dst_img[valid_mask] = src_img[src_x[valid_mask], src_y[valid_mask], :]

        return dst_img

    def point_transform(self, source, target, x, y):
        source_crs = Proj(source)
        target_crs = Proj(target)
        X_trans, Y_trans = transform(source_crs, target_crs, x, y)
        return X_trans, Y_trans

    def assign_spatial_reference(self, filepath, Xa, Ya, d):
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(3857)

        ds = gdal.Open(filepath, gdal.GA_Update)
        ds.SetGeoTransform([Xa, d, 0, Ya, 0, -d])
        ds.SetProjection(sr.ExportToWkt())
        ds = None

    def process_image(self, input_folder, output_folder):
        input_files = os.listdir(input_folder)

        for filename in input_files:
            name, ext = os.path.splitext(filename)
            input_file_path = os.path.join(input_folder, filename)
            pathin = input_file_path
            data = Get_Image_angle(pathin)

            B, L, H, omega_rad, phi_rad, kappa_rad = map(float, data)
            self.set_parameters(B, L, H, omega_rad, phi_rad, kappa_rad)

            src = cv2.imread(pathin)
            self.src_h, self.src_w, _ = src.shape

            self.x0 = 0
            self.y0 = 0
            self.x0 = self.x0 * self.s
            self.y0 = self.y0 * self.s

            self.Xs, self.Ys = self.point_transform(4490, 3857, B, L)
            self.Xs += self.xfix
            self.Ys += self.yfix

            pos1 = self.pos(-self.src_w / 2 * self.s, self.src_h / 2 * self.s)
            pos2 = self.pos(self.src_w / 2 * self.s, self.src_h / 2 * self.s)
            pos3 = self.pos(self.src_w / 2 * self.s, -self.src_h / 2 * self.s)
            pos4 = self.pos(-self.src_w / 2 * self.s, -self.src_h / 2 * self.s)

            self.Xmax = max(pos1[0], pos2[0], pos3[0], pos4[0])
            self.Ymax = max(pos1[1], pos2[1], pos3[1], pos4[1])
            self.Xmin = min(pos1[0], pos2[0], pos3[0], pos4[0])
            self.Ymin = min(pos1[1], pos2[1], pos3[1], pos4[1])
            Xa = self.Xmin
            Ya = self.Ymax

            size = 1
            row = (self.Ymax - self.Ymin) / self.d * size
            col = (self.Xmax - self.Xmin) / self.d * size
            Row = int(row)
            Col = int(col)
            dst_shape = (Row, Col)

            dst_img = self.nearest(src, dst_shape)

            new_filename = name + ".tif"
            cv2.imwrite(os.path.join(output_folder, new_filename), dst_img)

            self.assign_spatial_reference(os.path.join(output_folder, new_filename), Xa, Ya, self.d)

def get_args():
    parser = argparse.ArgumentParser(description='correct input images')
    parser.add_argument('--input_folder', '-img', metavar='INPUT', nargs='+', help='Filenames of input images',
                        default=r'D:\ForestChangeMOD\data\images')
    parser.add_argument('--input_shapefile', '-s', metavar='qianqi', nargs='+', help='Filenames of input images',
                        default=r'D:\ForestChangeMOD\data\forest\forest.shp')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images',
                        default=r'\ForestChangeMOD\data\out')
    return parser.parse_args()

if __name__ == '__main__':
    import argparse
    # 屏蔽特定的 FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning, module="pyproj")
    args = get_args()
    # input_folder = ' '.join(args.input_folder)
    # out_folder = ' '.join(args.output)
    # input_shapefile = ' '.join(args.input_shapefile)
    input_folder = args.input_folder
    out_folder = args.output
    input_shapefile = args.input_shapefile

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    image_corrector = ImageCorrection(f=8.8, s=0.00241)
    image_corrector.process_image(input_folder, os.path.join(out_folder,'correct'))

    # batch_process_tif_files(os.path.join(out_folder,'correct'), os.path.join(out_folder,'qianqi'), cut)
