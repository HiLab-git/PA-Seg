import re
import GeodisTK
import time
import os
import argparse
import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pymic.util.image_process import get_ND_bounding_box, crop_ND_volume_with_bounding_box, set_ND_volume_roi_with_bounding_box_range
from utilities.utils import create_logger
from medpy.metric.binary import precision, recall, dc, jc
from natsort import natsorted
import pandas as pd

PHASES = ['training', 'validation', 'inference']
# PHASES = ['validation']
METRICS = ["precision", "recall", "dice", "jaccard"]

def geodesic_distance_1d(I, S, spacing, lamb, iter):
    '''
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)
    
def check_elements(elements, amount, elem_list):
    if (not len(elements) == len(amount)) or (not len(elements) == len(elem_list)):
        raise ValueError("length of elements is not equal to length of amount")
    for elem in elem_list:
        if not elem in elements:
            raise ValueError("{0:d} is not in elements".format(elem))

def generate_geodesic_labels(image_arr, anno_arr, spacing, seed, opt):
    bbox_min, bbox_max = get_ND_bounding_box(anno_arr)
    I = crop_ND_volume_with_bounding_box(image_arr, bbox_min, bbox_max)
    I = np.asarray(I, np.float32)
    S = crop_ND_volume_with_bounding_box(anno_arr, bbox_min, bbox_max)
    S = np.asarray(S == seed, np.uint8)
    distance = geodesic_distance_1d(I, S, spacing, opt.geodesic_weight, 4)
    dist_max, dist_min = distance.max(), distance.min()

    geodesic_dist_arr = np.zeros_like(image_arr)
    geodesic_label_arr = np.zeros_like(anno_arr)
    geodesic_dist_arr = set_ND_volume_roi_with_bounding_box_range(geodesic_dist_arr, bbox_min, \
        bbox_max, np.asarray(distance, image_arr.dtype))
    geodesic_label_arr = set_ND_volume_roi_with_bounding_box_range(geodesic_label_arr, bbox_min, \
        bbox_max, np.asarray(np.where(distance < dist_max * opt.geodesic_threshold, 1, 2), anno_arr.dtype))
    # set minimum-value pixel as label 0
    lab0 = (image_arr == image_arr.min())
    geodesic_label_arr[lab0] = 0
    return geodesic_dist_arr, geodesic_label_arr

def main():
    opt = parsing_data()
    geodesic_dir = os.path.join(opt.path_geodesic, "weight{0:}_threshold{1:}".format(\
        opt.geodesic_weight, opt.geodesic_threshold))
    os.makedirs(geodesic_dir, exist_ok=True)
    logger = create_logger(geodesic_dir)
    geodesic_label_dir = os.path.join(geodesic_dir, "geodesic_label")
    os.makedirs(geodesic_label_dir, exist_ok=True)
    geodesic_distance_dir = os.path.join(geodesic_dir, "geodesic_distance")
    os.makedirs(geodesic_distance_dir, exist_ok=True)
    df_split = pd.read_csv(opt.dataset_split,header =None)
    list_file, list_phase = df_split[0].tolist(), df_split[1].tolist()
    # Logging hyperparameters
    logger.info("[INFO] Hyperparameters")
    logger.info("--dataset_split {0:}".format(opt.dataset_split))
    logger.info("--path_images {0:}".format(opt.path_images))
    logger.info("--image_postfix {0:}".format(opt.image_postfix))
    logger.info("--path_labels {0:}".format(opt.path_labels))
    logger.info("--label_postfix {0:}".format(opt.label_postfix))
    logger.info("--path_anno_7points {0:}".format(opt.path_anno_7points))
    logger.info("--anno_7points_postfix {0:}".format(opt.anno_7points_postfix))
    logger.info("--path_geodesic {0:}".format(opt.path_geodesic))
    logger.info("--geodesic_weight {0:}".format(opt.geodesic_weight))
    logger.info("--geodesic_threshold {0:}".format(opt.geodesic_threshold))

    mod_ext = "_{0:}.nii.gz".format(opt.image_postfix)
    label_ext = "_{0:}.nii.gz".format(opt.label_postfix)
    anno_ext = "_{0:}.nii.gz".format(opt.anno_7points_postfix)
    dict_scores = {}
    for phase in PHASES:
        dict_scores[phase] = {"name": []}
        for metric in METRICS:
            dict_scores[phase][metric] = []

    for subject, phase in zip(list_file, list_phase):
        if phase in PHASES:
            pass
        else:
            continue
        logger.info(subject)
        logger.info(phase)
        anno_path = os.path.join(opt.path_anno_7points, subject + anno_ext)
        anno_sitk = sitk.ReadImage(anno_path)
        anno_arr = sitk.GetArrayFromImage(anno_sitk)

        flair_path = os.path.join(opt.path_images, subject + mod_ext)
        flair_sitk = sitk.ReadImage(flair_path)
        flair_arr = sitk.GetArrayFromImage(flair_sitk)
        spacing_raw = flair_sitk.GetSpacing()
        spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]

        label_path = os.path.join(opt.path_labels, subject + label_ext)
        label_sitk = sitk.ReadImage(label_path)
        label_arr = sitk.GetArrayFromImage(label_sitk)

        elements = np.unique(anno_arr)
        amount = []
        for elem in elements:
            amount.append(np.sum(anno_arr == elem))
        logger.info(elements)
        logger.info(amount)
        if len(elements) == 3:
            # lab 0, 1, 2
            check_elements(elements, amount, [0, 1, 2])
            geodesic_dist_arr, geodesic_label_arr = generate_geodesic_labels(flair_arr, anno_arr, spacing, 2, opt)
            geodesic_dist_sitk = sitk.GetImageFromArray(geodesic_dist_arr)
            geodesic_dist_sitk.CopyInformation(flair_sitk)
            sitk.WriteImage(geodesic_dist_sitk, os.path.join(geodesic_distance_dir, subject + "_GeodesicDistance.nii.gz"))
            geodesic_label_sitk = sitk.GetImageFromArray(geodesic_label_arr)
            geodesic_label_sitk.CopyInformation(anno_sitk)
            sitk.WriteImage(geodesic_label_sitk, os.path.join(geodesic_label_dir, subject + "_GeodesicLabel.nii.gz"))
        elif len(elements) == 5:
            check_elements(elements, amount, [0, 1, 2, 3, 4])
            # lab 0, 1, 2
            anno_012 = np.where(np.logical_or(anno_arr == 0, np.logical_or(anno_arr == 1, anno_arr == 2)), anno_arr, 0)
            geodesic_dist_arr_012, geodesic_label_arr_012 = generate_geodesic_labels(flair_arr, anno_012, spacing, 2, opt)
            # lab 0, 3, 4
            anno_034 = np.where(np.logical_or(anno_arr == 0, np.logical_or(anno_arr == 3, anno_arr == 4)), anno_arr, 0)
            geodesic_dist_arr_034, geodesic_label_arr_034 = generate_geodesic_labels(flair_arr, anno_034, spacing, 4, opt)
            # merge two instance
            geodesic_label_arr = 2 * np.ones_like(anno_arr)
            union_lab1 = np.logical_or(geodesic_label_arr_012 == 1, geodesic_label_arr_034 == 1)
            intersection_lab0 = np.logical_and(geodesic_label_arr_012 == 0, geodesic_label_arr_034 == 0)
            geodesic_label_arr[union_lab1] = 1
            geodesic_label_arr[intersection_lab0] = 0

            geodesic_dist_sitk_012 = sitk.GetImageFromArray(geodesic_dist_arr_012)
            geodesic_dist_sitk_012.CopyInformation(flair_sitk)
            sitk.WriteImage(geodesic_dist_sitk_012, os.path.join(geodesic_distance_dir, subject + "_GeodesicDistance_012.nii.gz"))
            geodesic_label_sitk_012 = sitk.GetImageFromArray(geodesic_label_arr_012)
            geodesic_label_sitk_012.CopyInformation(anno_sitk)
            sitk.WriteImage(geodesic_label_sitk_012, os.path.join(geodesic_label_dir, subject + "_GeodesicLabel_012.nii.gz"))

            geodesic_dist_sitk_034 = sitk.GetImageFromArray(geodesic_dist_arr_034)
            geodesic_dist_sitk_034.CopyInformation(flair_sitk)
            sitk.WriteImage(geodesic_dist_sitk_034, os.path.join(geodesic_distance_dir, subject + "_GeodesicDistance_034.nii.gz"))
            geodesic_label_sitk_034 = sitk.GetImageFromArray(geodesic_label_arr_034)
            geodesic_label_sitk_034.CopyInformation(anno_sitk)
            sitk.WriteImage(geodesic_label_sitk_034, os.path.join(geodesic_label_dir, subject + "_GeodesicLabel_034.nii.gz"))

            geodesic_label_sitk = sitk.GetImageFromArray(geodesic_label_arr)
            geodesic_label_sitk.CopyInformation(anno_sitk)
            sitk.WriteImage(geodesic_label_sitk, os.path.join(geodesic_label_dir, subject + "_GeodesicLabel.nii.gz"))

        # calculate metrics
        dict_scores[phase]["name"].append(subject)
        for metric in METRICS:
            if metric == "precision":
                score = precision(geodesic_label_arr == 1, label_arr)
            elif metric == "recall":
                score = recall(geodesic_label_arr == 1, label_arr)
            elif metric == "dice":
                score = dc(geodesic_label_arr == 1, label_arr)
            elif metric == "jaccard":
                score = jc(geodesic_label_arr == 1, label_arr)
            else:
                raise ValueError
            dict_scores[phase][metric].append(score)
            logger.info("{0:}: {1:.4f}".format(metric, score))
            
    for phase in PHASES:
        dict_scores[phase]["name"].append("mean")
        dict_scores[phase]["name"].append("std")
        logger.info("-" * 6 + phase + "-" * 6)
        for metric in METRICS:
            mean_score = np.mean(dict_scores[phase][metric])
            std_score = np.std(dict_scores[phase][metric])
            dict_scores[phase][metric].append(mean_score)
            dict_scores[phase][metric].append(std_score)
            logger.info("{0:}  mean: {1:.4f} std: {2:.4f}".format(metric, mean_score, std_score))
        df_scores = pd.DataFrame(dict_scores[phase])
        df_scores.to_csv(os.path.join(geodesic_dir, "results_{0:}.csv".format(phase)))

def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to generate geodesic distance map and label as supervision")

    parser.add_argument("--dataset_split",
                    type=str,
                    default="splits/split_BraTS.csv",
                    help="Path to split file")

    parser.add_argument("--path_images",
                    type=str,
                    default="./data/BraTS/image/",
                    help="Path to the T2 scans")
    
    parser.add_argument("--image_postfix",
                    type=str,
                    default="Flair",
                    help="Postfix of the images")

    parser.add_argument("--path_labels",
                    type=str,
                    default="./data/BraTS/label/",
                    help="Path to the ground truth")

    parser.add_argument("--label_postfix",
                    type=str,
                    default="Label",
                    help="Postfix of the labels")

    parser.add_argument("--path_anno_7points",
                    type=str,
                    default="./data/BraTS/annotation_7points/",
                    help="Path to the annotations")

    parser.add_argument("--anno_7points_postfix", 
                    type=str, 
                    default="7points", 
                    help="Postfix of the 7point-annotations")

    parser.add_argument("--path_geodesic",
                    type=str,
                    default="./data/BraTS/geodesic/",
                    help="Path to save the geodesic labels and geodesic distance maps")

    parser.add_argument("--geodesic_weight",
                    type=float,
                    default=0.5,
                    help="Weight of spatial euclidean distance and image gradient")

    parser.add_argument("--geodesic_threshold",
                    type=float,
                    default=0.1,
                    help="Threshold of geodesic distance map")

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()
