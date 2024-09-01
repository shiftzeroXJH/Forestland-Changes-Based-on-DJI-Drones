import argparse
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask
import rasterio
from osgeo import gdal, ogr, osr

from distance import Fchange,resize_image
import warnings


# 设置 cudnn 的 Benchmark 和 deterministic 模式
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 全局禁用 NotGeoreferencedWarning 警告
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=r'.\ForestChangeMOD\checkpoints\unet.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images',
                        default=r'D:\odm_test\out')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.3,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_file_list(args):
    #提取输入和输出的文件夹路径
    input_folder = os.path.join(' '.join(args.input),'correct')
    output_folder = ' '.join(args.input)
    qianqi_folder = os.path.join(' '.join(args.input),'qianqi')
    # input_folder = os.path.join(args.input,'correct')
    # output_folder = args.input
    # qianqi_folder = os.path.join(args.input,'qianqi')

    os.makedirs(os.path.join(output_folder,'label'), exist_ok=True)

    #初始化列表存放每个文件输入输出的路径
    in_files = []
    out_files = []
    qianqi_files = []
    changemap_files = []

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            in_file = os.path.join(input_folder, filename)
            in_files.append(in_file)

            # 在每个输出文件路径后面添加_label
            out_filename_with_label = os.path.splitext(filename)[0] + '_label.tif'
            out_file = os.path.join(os.path.join(output_folder,'label'), out_filename_with_label)
            out_files.append(out_file)

            # 在前期数据文件夹中查找带有相同文件名的文件
            for qianqi_filename in os.listdir(qianqi_folder):
                if qianqi_filename.startswith(os.path.splitext(filename)[0]) and qianqi_filename.endswith('.tif'):
                    qianqi_file = os.path.join(qianqi_folder, qianqi_filename)
                    qianqi_files.append(qianqi_file)

            # 在每个输出文件路径后面添加_label
            changemap_with_label = os.path.splitext(filename)[0] + '_changemap.tif'
            changemap_file = os.path.join(output_folder, changemap_with_label)
            changemap_files.append(changemap_file)



    return in_files, out_files, qianqi_files, changemap_files


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def get_coordinate_reference_system_and_transform(tif_path):
    """
    从 TIFF 文件中获取坐标系信息
    """
    with rasterio.open(tif_path) as src:
        crs = src.crs
        transform = src.transform

    return crs, transform

def apply_coordinate_reference_system(source_tif_path, target_tif_path):
    """
    将源 TIFF 文件的坐标系信息应用到目标 TIFF 文件
    """
    source_crs, source_transform = get_coordinate_reference_system_and_transform(source_tif_path)

    # 读取目标 TIFF 文件
    with rasterio.open(target_tif_path, 'r+') as target:
        # 设置目标 TIFF 文件的坐标系信息
        target.crs = source_crs

        # 设置目标 TIFF 文件的变换信息
        target.transform = source_transform


def check_georeferencing(tif_path):
    with rasterio.open(tif_path) as src:
        if src.bounds is None:
            print("文件没有地理信息。")
        else:
            pass


def changemap(image1_path, image2_path, image3_path, output_path):
    #设置去除阴影阈值
    threshold = 20

    # 读取三个 TIFF 图像
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    img3 = Image.open(image3_path)

    # 确保三个图像大小相同
    if img1.size != img2.size or img1.size != img3.size:
        raise ValueError("三幅图像大小不同")

    # 将图像转换为 NumPy 数组以进行像素值相减
    array1 = np.array(img1)
    array2 = np.array(img2)
    array3 = np.array(img3)

    # 如果是彩色图像，只取第一个通道的值
    if len(array2.shape) == 3 and array2.shape[2] > 1:
        array2 = array2[:, :, 0]
    if len(array3.shape) == 3 and array3.shape[2] > 1:
        array3 = np.min(array3, axis=2)  # 对所有通道取最小值

    # 将所有值为255的元素替换为1
    array2[array2 == 255] = 1

    # 创建一个布尔掩码，指示array1中哪些元素不为0
    nonzero_mask = array1 != 0

    unique1 = np.unique(array1)
    unique2 = np.unique(array2)


    # 使用布尔掩码将array1中不为0的元素减去array2
    result = np.where(nonzero_mask, array1 - array2, array1)

    # 检查result为2的像素位置，在第三张图像中检查是否为0
    result = np.where((result == 2) & (array3 < threshold), 1, result)

    # 创建新的图像并保存
    subtracted_image = Image.fromarray(result.astype(np.uint8))
    subtracted_image.save(output_path)

    #赋予坐标系
    apply_coordinate_reference_system(image1_path, output_path)

    endpath = output_path[: -13] + "CM.shp"
    raster_to_vector(output_path,endpath)



def raster_to_vector(input_raster_path, output_shapefile_path):
    min_area_threshold = 100
    inraster = gdal.Open(input_raster_path)
    inband = inraster.GetRasterBand(1)
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())

    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shapefile_path):
        drv.DeleteDataSource(output_shapefile_path)
    Polygon = drv.CreateDataSource(output_shapefile_path)
    Poly_layer = Polygon.CreateLayer(os.path.basename(input_raster_path)[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)

    newField = ogr.FieldDefn('value', ogr.OFTReal)
    Poly_layer.CreateField(newField)

    # 计算面积并添加到属性中
    gdal.Polygonize(inband, None, Poly_layer, 0, ['8CONNECTED=8', 'MASK_BAND=0', '8CONNECTED=8'])
    area_field = ogr.FieldDefn('Area', ogr.OFTReal)
    Poly_layer.CreateField(area_field)
    for feature in Poly_layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()  # 获取面积
        feature.SetField('Area', area)
        Poly_layer.SetFeature(feature)

    # 清除属性过滤器
    Poly_layer.SetAttributeFilter(None)

    # 过滤掉像素值不为2的要素
    Poly_layer.SetAttributeFilter("value <> 2")
    for feature in Poly_layer:
        Poly_layer.DeleteFeature(feature.GetFID())

    # 根据面积阈值删除小的要素
    if min_area_threshold is not None:
        Poly_layer.SetAttributeFilter(None)  # 清除属性过滤器
        for feature in Poly_layer:
            area = feature.GetField('Area')
            if area < min_area_threshold:
                Poly_layer.DeleteFeature(feature.GetFID())

    Polygon.SyncToDisk()
    Polygon = None


def get_edge(image_path):
    # 读取图像并直接转换为灰度影像
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError("The image could not be loaded.")

    # 将不为0的像素变成255
    binary_image = np.where(gray_image != 0, 255, 0).astype(np.uint8)
    resized_change = resize_image(binary_image, 10)

    # 进行边缘检测
    edge = cv2.Canny(resized_change, 10, 100)

    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge.astype(np.uint8), connectivity=8)

    # 创建一个输出图像，只保留面积大于等于 100 的组件
    output_image = np.zeros_like(edge)
    for label in range(1, num_labels):  # 忽略背景
        # stats[label, 4] 是该组件的面积
        if stats[label, -1] >= 100:  # 只保留面积大于等于 100 的组件
            output_image[labels == label] = 255

    return output_image


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files, out_files, qianqi_files, changemap_files = get_file_list(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)


        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

        check_georeferencing(filename)
        apply_coordinate_reference_system(filename, out_filename)

        edge = get_edge(filename)
        qianqi = qianqi_files[i]
        label_path = out_filename
        Fchange(label_path,qianqi,edge)


