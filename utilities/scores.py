import argparse
import os
from tqdm import tqdm
from medpy.metric.binary import dc, jc, hd95, assd, precision, recall
import numpy as np
import nibabel as nib
import pandas as pd
from natsort import natsorted

METRICS = ["dice", "jaccard", "hd95", "assd", "precision", "recall"]

def main():
    opt = parsing_data()
    name_folder = os.path.join(opt.model_dir, opt.network)

    df_split = pd.read_csv(opt.dataset_split,header =None)
    list_patient = natsorted(df_split[df_split[1].isin([opt.phase])][0].tolist())
    
    # Logging hyperparameters
    print("[INFO] Hyperparameters")
    print("--model_dir {0:}".format(opt.model_dir))
    print("--network {0:}".format(opt.network))
    print("--dataset_split {0:}".format(opt.dataset_split))
    print("--image_postfix {0:}".format(opt.image_postfix))
    print("--phase {0:}".format(opt.phase))
    print("--path_labels {0:}".format(opt.path_labels))
    print("--label_postfix {0:}".format(opt.label_postfix))

    dict_scores = {"name": []}
    for metric in METRICS:
        dict_scores[metric] = []

    exception_list = []
    for patient in tqdm(list_patient):
        path_gt = os.path.join(opt.path_labels, f"{patient}_{opt.label_postfix}.nii.gz")
        path_pred = os.path.join(name_folder,'output_pred',f"{patient}_{opt.image_postfix}",f"{patient}_{opt.image_postfix}_seg.nii.gz")
        gt = nib.funcs.as_closest_canonical(nib.load(path_gt)).get_fdata().squeeze()
        pred = nib.funcs.as_closest_canonical(nib.load(path_pred)).get_fdata().squeeze()
        affine = nib.funcs.as_closest_canonical(nib.load(path_gt)).affine
        voxel_spacing = [abs(affine[k,k]) for k in range(3)]

        dict_scores["name"].append(patient)
        for metric in METRICS:
            if metric == "dice":
                score = dc(pred, gt)
            elif metric == "jaccard":
                score = jc(pred, gt)
            elif metric == "hd95":
                try:
                    score = hd95(pred, gt, voxel_spacing)
                except RuntimeError as e:
                    exception_list.append(patient)
                    # refresh dict_scores
                    dict_scores["name"].remove(patient)
                    for metric in METRICS:
                        dict_scores[metric] = dict_scores[metric][0:len(dict_scores["name"])]
                    break
            elif metric == "assd":
                try:
                    score = assd(pred, gt, voxel_spacing)
                except RuntimeError as e:
                    exception_list.append(patient)
                    dict_scores["name"].remove(patient)
                    # refresh dict_scores
                    for metric in METRICS:
                        dict_scores[metric] = dict_scores[metric][0:len(dict_scores["name"])]
                    break
            elif metric == "precision":
                score = precision(pred, gt)
            elif metric == "recall":
                score = recall(pred, gt)
            else:
                raise ValueError
            dict_scores[metric].append(score)

    print(name_folder)
    dict_scores["name"].append("mean")
    dict_scores["name"].append("std")
    for metric in METRICS:
        mean_score = np.mean(dict_scores[metric])
        std_score = np.std(dict_scores[metric])
        dict_scores[metric].append(mean_score)
        dict_scores[metric].append(std_score)
        print("{0:}  mean: {1:.4f} std: {2:.4f}".format(metric, mean_score, std_score))

    df_scores = pd.DataFrame(dict_scores)
    df_scores.to_csv(os.path.join(name_folder, "results_{0:}.csv".format(opt.phase)))
    if not exception_list:  # if exception_list is empty
        print("All cases pass!")
    else:
        print("Some cases raise RuntimeError when calculating hd95 and assd:")
        print(exception_list)

def parsing_data():
    parser = argparse.ArgumentParser(
        description='Computing scores')


    parser.add_argument('--model_dir',
                        type=str,
                        default="./models/debug/",
                        help="Path to the model directory")

    parser.add_argument("--network",
                    type=str,
                    default="U_Net2D5",
                    help="Network type")

    parser.add_argument("--dataset_split",
                        type=str,
                        default="./splits/split_inextremis_budget1.csv")

    parser.add_argument("--image_postfix",
                type=str,
                default="T2",
                help="Postfix of the images")

    parser.add_argument('--phase',
                        type=str,
                        default='inference')

    parser.add_argument("--path_labels",
                        type=str,
                        default="./data/VS/label_crop/")

    parser.add_argument("--label_postfix",
                        type=str,
                        default="Label",
                        help="Postfix of the labels")

    opt = parser.parse_args()

    return opt

if __name__ == '__main__':
    main()