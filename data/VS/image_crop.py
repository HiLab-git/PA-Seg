import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pymic.util.image_process import get_ND_bounding_box
from tqdm import tqdm
from monai.transforms import AddChannel, SpatialPad, SpatialCrop

def get_bounding_box_start(lab, bbox):
    D, H, W = lab.shape
    indxes  = np.nonzero(lab)
    d_min, d_max = indxes[0].min(), indxes[0].max()
    h_min, h_max = indxes[1].min(), indxes[1].max()
    w_min, w_max = indxes[2].min(), indxes[2].max() 
    d_cen = int((d_min + d_max) / 2)
    h_cen = int((h_min + h_max) / 2)
    w_cen = int((w_min + w_max) / 2)
    d0 = d_cen - int(bbox[0] / 2)
    h0 = h_cen - int(bbox[1] / 2)
    w0 = w_cen - int(bbox[2] / 2)
    d0 = min(max(0, d0), D - bbox[0])
    h0 = min(max(0, h0), H - bbox[1])
    w0 = min(max(0, w0), W - bbox[2])
    bbox_start = [d0, h0, w0]
    return bbox_start

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop VS images and labels')
    parser.add_argument('--data_dir', type=str, default="./data/VS/", help='(string) path to VS images and labels')
    parser.add_argument("--dataset_split", type=str, default="./splits/split_VS.csv", help="Path to split file")
    parser.add_argument("--image_postfix", type=str, default="T2", help="Postfix of the images")
    parser.add_argument("--label_postfix", type=str, default="Label", help="Postfix of the labels")
    args = parser.parse_args()

    image_dir = os.path.join(args.data_dir, "image")
    label_dir = os.path.join(args.data_dir, "label")
    image_crop_dir = os.path.join(args.data_dir, "image_crop")
    label_crop_dir = os.path.join(args.data_dir, "label_crop")
    assert os.path.isfile(args.dataset_split), print("[ERROR] Invalid split")
    df_split = pd.read_csv(args.dataset_split,header =None)
    list_file = df_split[0].tolist()
    print("patient number", len(list_file))
    mod_ext = "_{0:}.nii.gz".format(args.image_postfix)
    label_ext = "_{0:}.nii.gz".format(args.label_postfix)
    bbox = [48, 128, 128]

    max_fg_shape = [0, 0, 0]
    min_img_shape = [1000, 1000, 1000]
    for name in list_file:
        print(name)
        lab_name = os.path.join(label_dir, name + label_ext)
        lab_sitk = sitk.ReadImage(lab_name)
        lab_arr = sitk.GetArrayFromImage(lab_sitk)
        img_shape = lab_arr.shape
        print("image shape: {0:}".format(img_shape))
        for i, dim_size in enumerate(img_shape):
            if dim_size < min_img_shape[i]:
                min_img_shape[i] = dim_size

        bbox_min, bbox_max = get_ND_bounding_box(lab_arr)
        fg_shape = [j - i for i, j in zip(bbox_min, bbox_max)]
        print("foreground shape: {0:}".format(fg_shape))
        for i, dim_size in enumerate(fg_shape):
            if dim_size > max_fg_shape[i]:
                max_fg_shape[i] = dim_size
    print("-" * 12 + "-" * 16)
    print("minimum image shape: {0:}".format(min_img_shape))
    print("maxmum foreground shape: {0:}".format(max_fg_shape))

    channel_adder = AddChannel()
    padder = SpatialPad(bbox)
    os.makedirs(image_crop_dir, exist_ok=True)
    os.makedirs(label_crop_dir, exist_ok=True)
    for name in tqdm(list_file):
        img_name = os.path.join(image_dir, name + mod_ext)
        img_sitk = sitk.ReadImage(img_name)
        img_arr = sitk.GetArrayFromImage(img_sitk)
        img_arr = padder(channel_adder(img_arr))

        lab_name = os.path.join(label_dir, name + label_ext)
        lab_sitk = sitk.ReadImage(lab_name)
        lab_arr = sitk.GetArrayFromImage(lab_sitk)
        lab_arr = padder(channel_adder(lab_arr))

        bbox_start = get_bounding_box_start(lab_arr[0], bbox)
        bbox_end = [start+bbox[i] for i, start in enumerate(bbox_start)]
        cropper = SpatialCrop(roi_start=bbox_start, roi_end=bbox_end)
        img_crop_arr = cropper(img_arr)
        lab_crop_arr = cropper(lab_arr)

        img_crop_sitk = sitk.GetImageFromArray(img_crop_arr.squeeze())
        lab_crop_sitk = sitk.GetImageFromArray(lab_crop_arr.squeeze())
        img_crop_sitk.SetSpacing(img_sitk.GetSpacing())
        lab_crop_sitk.SetSpacing(lab_sitk.GetSpacing())
        img_crop_sitk.SetOrigin(img_sitk.GetOrigin())
        lab_crop_sitk.SetOrigin(lab_sitk.GetOrigin())
        img_crop_sitk.SetDirection(img_sitk.GetDirection())
        lab_crop_sitk.SetDirection(lab_sitk.GetDirection())
        sitk.WriteImage(img_crop_sitk, os.path.join(image_crop_dir, name + mod_ext))
        sitk.WriteImage(lab_crop_sitk, os.path.join(label_crop_dir, name + label_ext))
