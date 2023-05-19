#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from medpy.metric.binary import dc, hd95, precision, recall

from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadNiftid,
    AddChanneld,
    SpatialPadd,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    ToTensord,
)

from utilities.losses import DC, CE, DC_CE, DC_CE_Focal, PartialLoss, SizeLoss, VarianceLoss
from utilities.gated_crf_loss3d import ModelLossSemsegGatedCRF3D, ModelLossSemsegGatedCRF3D22D
from utilities.utils import (
    create_logger, 
    poly_lr, 
    infinite_iterable)
from utilities.ramps import sigmoid_rampup, linear_rampup, cosine_rampdown
# from utilities.geodesics import generate_geodesics

# from ScribbleDA.scribbleDALoss import CRFLoss 
from network.net_dict import get_network

# Define training and patches sampling parameters   
NB_CLASSES = 2
PHASES = ["training", "validation"]

# Training parameters
weight_decay = 3e-5

def train(paths_dict, model, transformation, criterion, device, save_path, fold_dir, logger, opt):
    
    since = time.time()
    writer = SummaryWriter(fold_dir)

    # Define transforms for data normalization and augmentation
    subjects_train = Dataset(
        paths_dict["training"], 
        transform=transformation["training"])

    subjects_val = Dataset(
        paths_dict["validation"], 
        transform=transformation["validation"])
    
    # Dataloaders
    dataloaders = dict()
    dataloaders["training"] = infinite_iterable(
        DataLoader(subjects_train, batch_size=opt.batch_size, num_workers=2, shuffle=True)
        )
    dataloaders["validation"] = infinite_iterable(
        DataLoader(subjects_val, batch_size=1, num_workers=2)
        )

    nb_batches = {
        "training": len(paths_dict["training"]) // opt.batch_size + 1, 
        "validation": len(paths_dict["validation"]) // 1
        }

    # Training parameters are saved 
    df_path = os.path.join(fold_dir,"log.csv")
    if os.path.isfile(df_path): # If the training already started
        df = pd.read_csv(df_path, index_col=False)
        epoch = df.iloc[-1]["epoch"]
        best_epoch = df.iloc[-1]["best_epoch"]
        initial_lr = df.iloc[-1]["lr"]
        best_val = df.iloc[-1]["best_val"]
        model.load_state_dict(torch.load(save_path.format("best")))

    else: # If training from scratch
        columns=["epoch", "best_epoch", "lr", "best_val"]
        df = pd.DataFrame(columns=columns)
        epoch = 0
        best_epoch = 0
        initial_lr = opt.learning_rate
        best_val = None


    # Optimisation policy mimicking nnUnet training policy
    optimizer = torch.optim.SGD(model.parameters(),  initial_lr, 
                weight_decay=weight_decay, momentum=0.99, nesterov=True)
                
    # GatedCRF Loss initialisation
    gated_crf_loss = ModelLossSemsegGatedCRF3D22D()
    # loss_gatedcrf_kernels_desc = [{"weight": 0.9, "xy": 6, "rgb": 0.1}, {"weight": 0.1, "xy": 6}]
    loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
    loss_gatedcrf_radius = opt.kernel_radius
    down_size = opt.down_size

    # Variance Loss initialisation
    variance_loss = VarianceLoss()

    # Training loop
    continue_training = True
    while continue_training:
        epoch+=1
        logger.info("-" * 10)
        logger.info("Epoch {}/".format(epoch))
        for param_group in optimizer.param_groups:
            logger.info("Current learning rate is: {}".format(param_group["lr"]))
            
        # Each epoch has a training and validation phase
        for phase in PHASES:
            if phase == "training":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode 

            # Initializing the statistics
            running_loss = 0.0
            running_loss_crf = 0.0
            running_loss_var = 0.0
            running_loss_seg = 0.0
            running_dice = 0.0
            epoch_samples = 0

            # Iterate over data
            for _ in tqdm(range(nb_batches[phase])):
                batch = next(dataloaders[phase])
                inputs = batch["img"].to(device) # T2 images
                labels = batch["label"].to(device)
                geodesic_labels = batch["geodesic_label"].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "training"):
                    if phase=="training": # Random patch predictions
                        outputs = model(inputs)
                    else:  # if validation, Inference on the full image
                        outputs = sliding_window_inference(
                            inputs=inputs,
                            roi_size=opt.spatial_shape,
                            sw_batch_size=1,
                            predictor=model,
                            mode="gaussian",
                        )
                        # Calculate Dice score
                        pred = outputs.detach().argmax(1, keepdim = True).squeeze().cpu().numpy()
                        gt = labels.detach().squeeze().cpu().numpy()
                        dice = dc(pred, gt)

                    # Segmentation loss
                    loss_seg = criterion(outputs, geodesic_labels, phase) 
                    # gated crf loss
                    if phase == "training":
                        outputs_down_HWD = F.interpolate(outputs, size=down_size, mode="trilinear", align_corners=True)
                        inputs_down_HWD = F.interpolate(inputs, size=down_size, mode="trilinear", align_corners=True)
                        # outputs_down_HWD = torch.nn.functional.pad(outputs_down_HWD, (8, 8))
                        # inputs_down_HWD = torch.nn.functional.pad(inputs_down_HWD, (8, 8))
                        loss_crf_HWD = gated_crf_loss(torch.softmax(outputs_down_HWD, dim=1), loss_gatedcrf_kernels_desc, \
                            loss_gatedcrf_radius, inputs_down_HWD, down_size[0], down_size[1], down_size[2])["loss"]
                        outputs_down_DHW = torch.permute(outputs_down_HWD, (0, 1, 4, 2, 3))
                        inputs_down_DHW = torch.permute(inputs_down_HWD, (0, 1, 4, 2, 3))
                        loss_crf_DHW = gated_crf_loss(torch.softmax(outputs_down_DHW, dim=1), loss_gatedcrf_kernels_desc, \
                            loss_gatedcrf_radius, inputs_down_DHW, down_size[2], down_size[0], down_size[1])["loss"]
                        outputs_down_WDH = torch.permute(outputs_down_HWD, (0, 1, 3, 4, 2))
                        inputs_down_WDH = torch.permute(inputs_down_HWD, (0, 1, 3, 4, 2))
                        loss_crf_WDH = gated_crf_loss(torch.softmax(outputs_down_WDH, dim=1), loss_gatedcrf_kernels_desc, \
                            loss_gatedcrf_radius, inputs_down_WDH, down_size[1], down_size[2], down_size[0])["loss"]
                        loss_crf = (loss_crf_HWD + loss_crf_DHW + loss_crf_WDH) / 3.0
                        loss_variance = variance_loss(outputs, inputs)
                        loss = loss_seg + sigmoid_rampup(epoch, opt.rampup_epochs) * (opt.weight_gatedcrf * loss_crf + opt.weight_variance * loss_variance)
                    else:
                        loss_crf = torch.Tensor([0]).to(device)
                        loss_variance = torch.Tensor([0]).to(device)
                        loss = loss_seg + loss_crf + loss_variance

                    if phase == "training":
                        loss.backward()
                        optimizer.step()
                
                # Iteration statistics
                epoch_samples += 1
                running_loss += loss.item()
                running_loss_seg += loss_seg.item()
                running_loss_crf += loss_crf.item()
                running_loss_var += loss_variance.item()
                if phase == "validation":
                    running_dice += dice

            # Epoch statistcs
            epoch_loss = running_loss / epoch_samples
            epoch_loss_seg = running_loss_seg / epoch_samples
            epoch_loss_crf = running_loss_crf / epoch_samples
            epoch_loss_var = running_loss_var / epoch_samples
            epoch_dice = running_dice / epoch_samples

            writer.add_scalar("{0:}/loss".format(phase), epoch_loss, epoch)
            writer.add_scalar("{0:}/loss_seg".format(phase), epoch_loss_seg, epoch)
            writer.add_scalar("{0:}/loss_Crf".format(phase), epoch_loss_crf, epoch)
            writer.add_scalar("{0:}/loss_var".format(phase), epoch_loss_var, epoch)
            if phase == "validation":
                writer.add_scalar("{0:}/dice".format(phase), epoch_dice, epoch)
            if phase == "training":
                num_slice = inputs.shape[4]
                center_slice = num_slice // 2
                img = inputs[0, 0:1, :, :, [center_slice-10, center_slice, center_slice+10]].permute( \
                    3,0,1,2).repeat(1,3,1,1)
                img_grid = make_grid(img, 3, normalize=True)
                writer.add_image("img", img_grid, epoch)
                lab = labels[0, 0:1, :, :, [center_slice-10, center_slice, center_slice+10]].permute( \
                    3,0,1,2).repeat(1,3,1,1)
                lab_grid = make_grid(lab, 3, normalize=True)
                writer.add_image("label", lab_grid, epoch)
                geo_lab = geodesic_labels[0, 0:1, :, :, [center_slice-10, center_slice, center_slice+10]].permute( \
                    3,0,1,2).repeat(1,3,1,1)
                geo_lab_grid = make_grid(geo_lab, 3, normalize=True)
                writer.add_image("geodesic_label", geo_lab_grid, epoch)
                pred = outputs.argmax(dim=1, keepdim=True).float()
                pred = pred[0, 0:1, :, :, [center_slice-10, center_slice, center_slice+10]].permute( \
                    3,0,1,2).repeat(1,3,1,1)
                pred_grid = make_grid(pred, 3, normalize=True)
                writer.add_image("prediction", pred_grid, epoch)
           
            logger.info("{}  Loss Seg: {:.4f}".format(
                phase, epoch_loss_seg))
            logger.info("{}  Loss Crf: {:.4f}".format(
                phase, epoch_loss_crf))
            logger.info("{}  Loss Var: {:.4f}".format(
                phase, epoch_loss_var))
            logger.info("{}  Loss: {:.4f}".format(
                phase, epoch_loss))
            if phase == "validation":
                logger.info("{}  Dice: {:.4f}".format(phase, epoch_dice))
                
            # Saving best model on the validation set
            if phase == "validation":
                if best_val is None: # first iteration
                    best_val = epoch_dice
                    torch.save(model.state_dict(), save_path.format("best"))

                if epoch_dice > best_val:
                    best_val = epoch_dice
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path.format("best"))

                df = df.append(
                    {"epoch":epoch,
                    "best_epoch":best_epoch,
                    "best_val":best_val,  
                    "lr":param_group["lr"],}, 
                    ignore_index=True)
                df.to_csv(df_path, index=False)

                optimizer.param_groups[0]["lr"] = poly_lr(epoch, opt.max_epochs, opt.learning_rate, 0.9)

        if epoch == opt.max_epochs:
            torch.save(model.state_dict(), save_path.format("final"))
            continue_training=False
    
    time_elapsed = time.time() - since
    logger.info("[INFO] Training completed in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(f"[INFO] Best validation epoch is {best_epoch}")


def main():
    set_determinism(seed=19961216)
    opt = parsing_data()
        
    # FOLDERS
    fold_dir = os.path.join(opt.model_dir, opt.network)
    fold_dir_model = os.path.join(fold_dir,"models")
    if not os.path.exists(fold_dir_model):
        os.makedirs(fold_dir_model)
    save_path = os.path.join(fold_dir_model,"CP_{}.pth")

    logger = create_logger(fold_dir)
    logger.info("[INFO] Hyperparameters")
    logger.info("--model_dir {0:}".format(opt.model_dir))
    logger.info("--network {0:}".format(opt.network))
    logger.info("--batch_size {0:}".format(opt.batch_size))
    logger.info("--max_epochs {0:}".format(opt.max_epochs))
    logger.info("--rampup_epochs {0:}".format(opt.rampup_epochs))
    logger.info("--dataset_split {0:}".format(opt.dataset_split))
    logger.info("--path_images {0:}".format(opt.path_images))
    logger.info("--image_postfix {0:}".format(opt.image_postfix))
    logger.info("--path_labels {0:}".format(opt.path_labels))
    logger.info("--label_postfix {0:}".format(opt.label_postfix))
    logger.info("--path_geodesic_labels {0:}".format(opt.path_geodesic_labels))
    logger.info("--geodesic_label_postfix {0:}".format(opt.geodesic_label_postfix))
    logger.info("--learning_rate {0:}".format(opt.learning_rate))
    logger.info("--spatial_shape {0:}".format(opt.spatial_shape))
    logger.info("--weight_gatedcrf {0:}".format(opt.weight_gatedcrf))
    logger.info("--down_size {0:}".format(opt.down_size))
    logger.info("--kernel_radius {0:}".format(opt.kernel_radius))
    logger.info("--weight_variance {0:}".format(opt.weight_variance))

    # GPU CHECKING
    if torch.cuda.is_available():
        logger.info("[INFO] GPU available.")
        device = torch.device("cuda:0")
    else:
        raise logger.error(
            "[INFO] No GPU found")

    # SPLIT
    assert os.path.isfile(opt.dataset_split), logger.error("[ERROR] Invalid split")
    df_split = pd.read_csv(opt.dataset_split,header =None)
    list_file = dict()
    for split in PHASES:
        list_file[split] = df_split[df_split[1].isin([split])][0].tolist()


    # CREATING DICT FOR CACHEDATASET
    mod_ext = "_{0:}.nii.gz".format(opt.image_postfix)
    label_ext = "_{0:}.nii.gz".format(opt.label_postfix)
    geodesic_label_ext = "_{0:}.nii.gz".format(opt.geodesic_label_postfix)
    paths_dict = {split:[] for split in PHASES}

    for split in PHASES:
        for subject in list_file[split]:
            subject_data = dict()

            img_path = os.path.join(opt.path_images,subject+mod_ext)
            lab_path = os.path.join(opt.path_labels,subject+label_ext)
            geodesic_label_path = os.path.join(opt.path_geodesic_labels,subject+geodesic_label_ext)

            if os.path.exists(img_path) and os.path.exists(lab_path):
                subject_data["img"] = img_path
                subject_data["label"] = lab_path
                subject_data["geodesic_label"] = geodesic_label_path
                paths_dict[split].append(subject_data)
                
        logger.info(f"Nb patients in {split} data: {len(paths_dict[split])}")
            

    # PREPROCESSING
    transforms = dict()
    all_keys = ["img", "label", "geodesic_label"]

    transforms_training = (
        LoadNiftid(keys=all_keys),
        AddChanneld(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        SpatialPadd(keys=all_keys, spatial_size=opt.spatial_shape),
        RandFlipd(keys=all_keys, prob=0.2, spatial_axis=0),
        RandFlipd(keys=all_keys, prob=0.2, spatial_axis=1),
        RandFlipd(keys=all_keys, prob=0.2, spatial_axis=2),
        RandSpatialCropd(keys=all_keys, roi_size=opt.spatial_shape, random_center=True, random_size=False),
        ToTensord(keys=all_keys),
        )   
    transforms["training"] = Compose(transforms_training)   

    transforms_validation = (
        LoadNiftid(keys=all_keys),
        AddChanneld(keys=all_keys),
        Orientationd(keys=all_keys, axcodes="RAS"),
        NormalizeIntensityd(keys=["img"]),
        ToTensord(keys=all_keys)
        )
    transforms["validation"] = Compose(transforms_validation) 
 
    # MODEL
    logger.info("[INFO] Building model")

    model = get_network(opt.network, input_channels=1, output_channels=NB_CLASSES).to(device)
  
    logger.info("[INFO] Training")
    dice_ce = DC_CE(NB_CLASSES)
    criterion = PartialLoss(dice_ce)
    
    train(paths_dict, 
        model, 
        transforms, 
        criterion, 
        device, 
        save_path,
        fold_dir,
        logger,
        opt)

def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using geodesic labels as supervision")

    parser.add_argument("--model_dir",
                    type=str,
                    default="./models/debug/",
                    help="Path to the model directory")
    
    parser.add_argument("--network",
                    type=str,
                    default="U_Net2D5",
                    help="Network type")

    parser.add_argument("--batch_size",
                    type=int,
                    default=6,
                    help="Size of the batch size (default: 6)")

    parser.add_argument("--max_epochs",
                    type=int,
                    default=300,
                    help="Maximum epochs for training model")

    parser.add_argument("--rampup_epochs",
                    type=int,
                    default=30,
                    help="rampup epochs for regularization loss")
    
    parser.add_argument("--dataset_split",
                    type=str,
                    default="./splits/split_VS.csv",
                    help="Path to split file")
    
    parser.add_argument("--path_images",
                    type=str,
                    default="./data/VS/image_crop/",
                    help="Path to the T2 scans")

    parser.add_argument("--image_postfix",
                    type=str,
                    default="T2",
                    help="Postfix of the images")

    parser.add_argument("--path_labels",
                    type=str,
                    default="./data/VS/label_crop/",
                    help="Path to the extreme points")

    parser.add_argument("--label_postfix",
                    type=str,
                    default="Label",
                    help="Postfix of the labels")

    parser.add_argument("--path_geodesic_labels",
                    type=str,
                    default="./data/VS/geodesic/weight0.5_threshold0.2/geodesic_label/",
                    help="Path to the extreme points")

    parser.add_argument("--geodesic_label_postfix",
                    type=str,
                    default="GeodesicLabel",
                    help="Postfix of the geodesic labels")

    parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-2,
                    help="Initial learning rate")
    
    parser.add_argument("--spatial_shape",
                    type=int,
                    nargs="+",
                    default=(128,128,48),
                    help="Size of the window patch")

    parser.add_argument("--weight_gatedcrf",
                    type=float,
                    default=0.1)

    parser.add_argument("--down_size",
                    type=int,
                    nargs="+",
                    default=(64, 64, 48),
                    help="Downsample size before calculate crfloss for saving memory")

    parser.add_argument("--kernel_radius", 
                    type=int, 
                    nargs="+", 
                    default=(5, 5, 3),
                    help="loss_gatedcrf_radius")

    parser.add_argument("--weight_variance",
                    type=float,
                    default=0.1)

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    main()



