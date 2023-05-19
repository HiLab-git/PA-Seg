import argparse
import os
import SimpleITK as sitk
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge labels and move to required directory')
    parser.add_argument('--original_dir', type=str, default="./data/BraTS2019/MICCAI_BraTS_2019_Data_Training/")
    parser.add_argument("--destination_dir", type=str, default="./data/BraTS/")
    args = parser.parse_args()
    dict_split = {"name": [], "phase": []}
    image_dir = os.path.join(args.destination_dir, "image")
    label_dir = os.path.join(args.destination_dir, "label")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    for folder in ["HGG", "LGG"]:
        input_dir = os.path.join(args.original_dir, folder)
        sub_dir_list = os.listdir(input_dir)
        for sub_dir in sub_dir_list:
            print(sub_dir)
            dict_split["name"].append(sub_dir)

            label_name = os.path.join(input_dir, sub_dir, "{0:}_seg.nii.gz".format(sub_dir))
            label_sitk = sitk.ReadImage(label_name)
            label_arr = sitk.GetArrayFromImage(label_sitk)
            img_shape = label_arr.shape
            assert img_shape == (155, 240, 240)

            image_name = os.path.join(input_dir, sub_dir, "{0:}_flair.nii.gz".format(sub_dir))
            image_sitk = sitk.ReadImage(image_name)
            sitk.WriteImage(image_sitk, os.path.join(image_dir, sub_dir + "_Flair.nii.gz"))

            label_new_arr = np.where(label_arr == 0, 0, 1)
            label_new_sitk = sitk.GetImageFromArray(label_new_arr)
            label_new_sitk.CopyInformation(label_sitk)
            sitk.WriteImage(label_new_sitk, os.path.join(label_dir, sub_dir + "_Label.nii.gz"))
