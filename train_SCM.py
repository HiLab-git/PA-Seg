#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import copy
import os
from torch.nn.modules import loss
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

from utilities.losses import DC, DC_CE, DC_CE_Focal, PartialLoss, KDLoss
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

def train(paths_dict, model1, model2, transformation, criterion, device, save_path1, save_path2, logger, opt):
    
    since = time.time()
    writer = SummaryWriter(opt.model_dir)

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
    df_path = os.path.join(opt.model_dir,"log.csv")
    if os.path.isfile(df_path): # If the training already started
        df = pd.read_csv(df_path, index_col=False)
        epoch = df.iloc[-1]["epoch"]
        initial_lr = df.iloc[-1]["lr"]
        best_epoch1 = df.iloc[-1]["best_epoch1"]
        best_val1 = df.iloc[-1]["best_val1"]
        model1.load_state_dict(torch.load(save_path1.format("best")))
        best_epoch2 = df.iloc[-1]["best_epoch2"]
        best_val2 = df.iloc[-1]["best_val2"]
        model2.load_state_dict(torch.load(save_path2.format("best")))

    else: # If training from scratch
        columns=["epoch", "lr", "best_epoch1", "best_val1", "best_epoch2", "best_val2"]
        df = pd.DataFrame(columns=columns)
        epoch = 0
        initial_lr = opt.learning_rate
        best_epoch1 = 0
        best_val1 = None
        best_epoch2 = 0
        best_val2 = None


    # Optimisation policy mimicking nnUnet training policy
    optimizer1 = torch.optim.SGD(model1.parameters(),  initial_lr, 
                weight_decay=weight_decay, momentum=0.99, nesterov=True)
    optimizer2 = torch.optim.SGD(model2.parameters(),  initial_lr, 
                weight_decay=weight_decay, momentum=0.99, nesterov=True)

    # Knowledge Distillation Loss initialisation
    loss_kd = KDLoss(opt.T)

    # Models for genetating pseudo labels
    model1_pseudo = copy.deepcopy(model1)
    model1_pseudo.eval()
    model2_pseudo = copy.deepcopy(model2)
    model2_pseudo.eval()

    # Training loop
    continue_training = True
    while continue_training:        
        
        epoch+=1
        logger.info("-" * 10)
        logger.info("Epoch {}/".format(epoch))
        for param_group1, param_group2 in zip(optimizer1.param_groups, optimizer2.param_groups):
            logger.info("Current learning rate is: {} {}".format(param_group1["lr"], param_group2["lr"]))
            
        # Each epoch has a training and validation phase
        for phase in PHASES:
            if phase == "training":
                model1.train()  # Set model to training mode
                model2.train()
            else:
                model1.eval()  # Set model to evaluate mode 
                model2.eval()

            # Initializing the statistics
            running_loss1 = 0.0
            running_loss_seg1 = 0.0
            running_loss_kd1 = 0.0
            running_dice1 = 0.0
            running_loss2 = 0.0
            running_loss_seg2 = 0.0
            running_loss_kd2 = 0.0
            running_dice2 = 0.0
            epoch_samples = 0

            # Iterate over data
            for _ in tqdm(range(nb_batches[phase])):
                batch = next(dataloaders[phase])
                inputs = batch["img"].to(device) # T2 images
                labels = batch["label"].to(device)

                # zero the parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                with torch.set_grad_enabled(phase == "training"):
                    if phase=="training": # Random patch predictions
                        outputs1 = model1(inputs)
                        outputs2 = model2(inputs)
                        outputs1_pseudo = model1_pseudo(inputs)
                        outputs2_pseudo = model2_pseudo(inputs)
                    else:  # if validation, Inference on the full image
                        outputs1 = sliding_window_inference(
                            inputs=inputs,
                            roi_size=opt.spatial_shape,
                            sw_batch_size=1,
                            predictor=model1,
                            mode="gaussian",
                        )
                        outputs2 = sliding_window_inference(
                            inputs=inputs,
                            roi_size=opt.spatial_shape,
                            sw_batch_size=1,
                            predictor=model2,
                            mode="gaussian",
                        )
                        
                    pred1 = outputs1.detach().argmax(1, keepdim=True)
                    pred2 = outputs2.detach().argmax(1, keepdim=True)

                    # Calculate Dice score
                    if phase == "validation":
                        dice1 = dc(pred1.cpu().numpy(), labels.detach().cpu().numpy())
                        dice2 = dc(pred2.cpu().numpy(), labels.detach().cpu().numpy())

                    if phase == "training":
                        pred1_pseudo = outputs1_pseudo.detach().argmax(1, keepdim=True)
                        pred2_pseudo = outputs2_pseudo.detach().argmax(1, keepdim=True)
                        # Segmentation loss and knowledge distillation loss
                        loss_seg1 = criterion(outputs1, pred1_pseudo, phase)
                        loss_kd1 = loss_kd(outputs1.permute(0, 2, 3, 4, 1).reshape(-1, NB_CLASSES), 
                                        outputs2_pseudo.detach().permute(0, 2, 3, 4, 1).reshape(-1, NB_CLASSES))
                        loss1 = (1 - opt.weight_kd) * loss_seg1 + opt.weight_kd * loss_kd1

                        loss_seg2 = criterion(outputs2, pred2_pseudo, phase)
                        loss_kd2 = loss_kd(outputs2.permute(0, 2, 3, 4, 1).reshape(-1, NB_CLASSES), 
                                        outputs1_pseudo.detach().permute(0, 2, 3, 4, 1).reshape(-1, NB_CLASSES))
                        loss2 = (1 - opt.weight_kd) * loss_seg2 + opt.weight_kd * loss_kd2
                        loss1.backward()
                        optimizer1.step()
                        loss2.backward()
                        optimizer2.step()
                    else:
                        loss_seg1 = torch.Tensor([0]).to(device)
                        loss_kd1 = torch.Tensor([0]).to(device)
                        loss1 = torch.Tensor([0]).to(device)
                        loss_seg2 = torch.Tensor([0]).to(device)
                        loss_kd2 = torch.Tensor([0]).to(device)
                        loss2 = torch.Tensor([0]).to(device)
                
                # Iteration statistics
                epoch_samples += 1
                running_loss1 += loss1.item()
                running_loss_seg1 += loss_seg1.item()
                running_loss_kd1 += loss_kd1.item()
                running_loss2 +=loss2.item()
                running_loss_seg2 += loss_seg2.item()
                running_loss_kd2 += loss_kd2.item()
                if phase == "validation":
                    running_dice1 += dice1
                    running_dice2 += dice2

            # Epoch statistcs
            epoch_loss1 = running_loss1 / epoch_samples
            epoch_loss_seg1 = running_loss_seg1 / epoch_samples
            epoch_loss_kd1 = running_loss_kd1 / epoch_samples
            epoch_dice1 = running_dice1 / epoch_samples
            epoch_loss2 = running_loss2 / epoch_samples
            epoch_loss_seg2 = running_loss_seg2 / epoch_samples
            epoch_loss_kd2 = running_loss_kd2 / epoch_samples
            epoch_dice2 = running_dice2 / epoch_samples

            writer.add_scalar("{0:}/loss1".format(phase), epoch_loss1, epoch)
            writer.add_scalar("{0:}/loss_seg1".format(phase), epoch_loss_seg1, epoch)
            writer.add_scalar("{0:}/loss_kd1".format(phase), epoch_loss_kd1, epoch)
            writer.add_scalar("{0:}/loss2".format(phase), epoch_loss2, epoch)
            writer.add_scalar("{0:}/loss_seg2".format(phase), epoch_loss_seg2, epoch)
            writer.add_scalar("{0:}/loss_kd2".format(phase), epoch_loss_kd2, epoch)
            
            if phase == "validation":
                writer.add_scalar("{0:}/dice1".format(phase), epoch_dice1, epoch)
                writer.add_scalar("{0:}/dice2".format(phase), epoch_dice2, epoch)
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

                pred1 = pred1[0, 0:1, :, :, [center_slice-10, center_slice, center_slice+10]].permute( \
                    3,0,1,2).repeat(1,3,1,1)
                pred_grid1 = make_grid(pred1.float(), 3, normalize=True)
                writer.add_image("prediction1", pred_grid1, epoch)

                pred2 = pred2[0, 0:1, :, :, [center_slice-10, center_slice, center_slice+10]].permute( \
                    3,0,1,2).repeat(1,3,1,1)
                pred_grid2 = make_grid(pred2.float(), 3, normalize=True)
                writer.add_image("prediction2", pred_grid2, epoch)
           
            logger.info("{}  Loss Seg1: {:.4f}".format(
                phase, epoch_loss_seg1))
            logger.info("{}  Loss KD1: {:.4f}".format(
                phase, epoch_loss_kd1))
            logger.info("{}  Loss1: {:.4f}".format(
                phase, epoch_loss1))
            logger.info("{}  Loss Seg2: {:.4f}".format(
                phase, epoch_loss_seg2))
            logger.info("{}  Loss KD2: {:.4f}".format(
                phase, epoch_loss_kd2))
            logger.info("{}  Loss2: {:.4f}".format(
                phase, epoch_loss2))
            if phase == "validation":
                logger.info("{}  Dice1: {:.4f}".format(phase, epoch_dice1))
                logger.info("{}  Dice2: {:.4f}".format(phase, epoch_dice2))
                
            # Saving best model on the validation set
            if phase == "validation":
                if (best_val1 is None) or (epoch_dice1 > best_val1):
                    best_val1 = epoch_dice1
                    best_epoch1 = epoch
                    torch.save(model1.state_dict(), save_path1.format("best"))

                if (best_val2 is None) or (epoch_dice2 > best_val2) :
                    best_val2 = epoch_dice2
                    best_epoch2 = epoch
                    torch.save(model2.state_dict(), save_path2.format("best"))

                df = df.append(
                    {"epoch":epoch,
                    "lr":param_group1["lr"],
                    "best_epoch1":best_epoch1,
                    "best_val1":best_val1,
                    "best_epoch2":best_epoch2,
                    "best_val2":best_val2,}, 
                    ignore_index=True)
                df.to_csv(df_path, index=False)

                optimizer1.param_groups[0]["lr"] = poly_lr(epoch, opt.max_epochs, opt.learning_rate, 0.9)
                optimizer2.param_groups[0]["lr"] = poly_lr(epoch, opt.max_epochs, opt.learning_rate, 0.9)
        
        # Iterative training
        if epoch % opt.iterative_epochs == 0:
            model1_pseudo.load_state_dict(torch.load(save_path1.format("best")))
            model1_pseudo.eval()
            model2_pseudo.load_state_dict(torch.load(save_path2.format("best")))
            model2_pseudo.eval()

        if epoch == opt.max_epochs:
            torch.save(model1.state_dict(), save_path1.format("final"))
            torch.save(model2.state_dict(), save_path2.format("final"))
            continue_training=False
    
    time_elapsed = time.time() - since
    logger.info("[INFO] Training completed in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info(f"[INFO] Best validation epoch of model1 is {best_epoch1}")
    logger.info(f"[INFO] Best validation epoch of model2 is {best_epoch2}")


def main():
    set_determinism(seed=19961216)
    opt = parsing_data()
        
    # FOLDERS
    fold_dir = opt.model_dir
    fold_dir_model1 = os.path.join(fold_dir, opt.network1,"models")
    if not os.path.exists(fold_dir_model1):
        os.makedirs(fold_dir_model1, exist_ok=True)
    save_path1 = os.path.join(fold_dir_model1,"CP_{}_model1.pth")

    fold_dir_model2 = os.path.join(fold_dir, opt.network2,"models")
    if not os.path.exists(fold_dir_model2):
        os.makedirs(fold_dir_model2, exist_ok=True)
    save_path2 = os.path.join(fold_dir_model2,"CP_{}_model2.pth")

    logger = create_logger(fold_dir)
    logger.info("[INFO] Hyperparameters")
    logger.info("--pretrained_model1_dir {0:}".format(opt.pretrained_model1_dir))
    logger.info("--network1 {0:}".format(opt.network1))
    logger.info("--pretrained_model2_dir {0:}".format(opt.pretrained_model2_dir))
    logger.info("--network2 {0:}".format(opt.network2))
    logger.info("--model_dir {0:}".format(opt.model_dir))
    logger.info("--batch_size {0:}".format(opt.batch_size))
    logger.info("--max_epochs {0:}".format(opt.max_epochs))
    logger.info("--iterative_epochs {0:}".format(opt.iterative_epochs))
    logger.info("--dataset_split {0:}".format(opt.dataset_split))
    logger.info("--path_images {0:}".format(opt.path_images))
    logger.info("--image_postfix {0:}".format(opt.image_postfix))
    logger.info("--path_labels {0:}".format(opt.path_labels))
    logger.info("--label_postfix {0:}".format(opt.label_postfix))
    logger.info("--learning_rate {0:}".format(opt.learning_rate))
    logger.info("--spatial_shape {0:}".format(opt.spatial_shape))
    logger.info("--weight_kd {0:}".format(opt.weight_kd))
    logger.info("--T {0:}".format(opt.T))

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
    paths_dict = {split:[] for split in PHASES}

    for split in PHASES:
        for subject in list_file[split]:
            subject_data = dict()

            img_path = os.path.join(opt.path_images,subject+mod_ext)
            lab_path = os.path.join(opt.path_labels,subject+label_ext)

            if os.path.exists(img_path) and os.path.exists(lab_path):
                subject_data["img"] = img_path
                subject_data["label"] = lab_path
                paths_dict[split].append(subject_data)
                
        logger.info(f"Nb patients in {split} data: {len(paths_dict[split])}")
            

    # PREPROCESSING
    transforms = dict()
    all_keys = ["img", "label"]

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
    model1 = get_network(opt.network1, input_channels=1, output_channels=NB_CLASSES).to(device)
    model1.load_state_dict(torch.load(os.path.join(opt.pretrained_model1_dir, opt.network1, "models", "CP_best.pth")))

    model2 = get_network(opt.network2, input_channels=1, output_channels=NB_CLASSES).to(device)
    model2.load_state_dict(torch.load(os.path.join(opt.pretrained_model2_dir, opt.network2, "models", "CP_best.pth")))

    logger.info("[INFO] Training")
    dice_ce = DC_CE(NB_CLASSES)
    criterion = lambda pred, grnd, phase: dice_ce(pred, grnd)
    
    train(paths_dict, 
        model1,
        model2, 
        transforms, 
        criterion, 
        device, 
        save_path1,
        save_path2,
        logger,
        opt)

def parsing_data():
    parser = argparse.ArgumentParser(
        description="Script to train the models using geodesic labels as supervision")

    parser.add_argument("--pretrained_model1_dir",
                    type=str,
                    default="./models/VS/gatedcrfloss3d22d_multiview_varianceloss/",
                    help="Path to the pre-trained model1 directory")

    parser.add_argument("--network1",
                    type=str,
                    default="U_Net2D5",
                    help="Network type of model1")

    parser.add_argument("--pretrained_model2_dir",
                    type=str,
                    default="./models/VS/gatedcrfloss3d22d_multiview_varianceloss/",
                    help="Path to the pre-trained model1 directory")

    parser.add_argument("--network2",
                    type=str,
                    default="AttU_Net",
                    help="Network type of model2")

    parser.add_argument("--model_dir",
                    type=str,
                    default="./models/debug/",
                    help="Path to the model directory")

    parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="Size of the batch size (default: 6)")

    parser.add_argument("--max_epochs",
                    type=int,
                    default=300,
                    help="Maximum epochs for training model")

    parser.add_argument("--iterative_epochs",
                    type=int,
                    default=10,
                    help="Fix model1 and model2 periodically (default: 10)")
    
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

    parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-3,
                    help="Initial learning rate")
    
    parser.add_argument("--spatial_shape",
                    type=int,
                    nargs="+",
                    default=(128,128,48),
                    help="Size of the window patch")

    parser.add_argument("--weight_kd",
                    type=float,
                    default=0.5,
                    help="Weight of knowledge distillation loss")

    parser.add_argument('--T', 
                    type=float, 
                    default=4.0, 
                    help='temperature for knowledge distillation')

    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    main()



