# Adapted from: https://github.com/KCL-BMEIS/VS_Seg/blob/master/preprocessing/TCIA_data_convert_into_convenient_folder_structure.py

#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob
from natsort import natsorted
import pydicom
import SimpleITK as sitk
import re
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Convert the T2 scans from the TCIA dataset into the Nifti format')
    parser.add_argument('--input', type=str, default="./data/manifest-T2/Vestibular-Schwannoma-SEG", help='(string) path to TCIA dataset, in "Descriptive Directory Name" format, for example /home/user/.../manifest-T2/Vestibular-Schwannoma-SEG')
    parser.add_argument('--output', type=str, default="./data/VS/image", help='(string) path to output folder')
    args = parser.parse_args()

    input_path = args.input 
    output_path = args.output 

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    cases = natsorted(glob(os.path.join(input_path, '*')))

    for case in tqdm(cases):
        if "LICENSE" in case:
            continue
        folders = glob(case+'/*/*')
        
        MRs = [] 
        MRs_paths = []
        
        # Test that the DICOM is a MRI
        for folder in folders:
            first_file = glob(folder+"/*")[0]
            dd = pydicom.read_file(first_file)

            if dd['Modality'].value == 'MR':
                MRs.append(dd)
                MRs_paths.append(first_file)
            
            else:
                raise Exception
            
        file_paths = None
        # Test that the DICOM is a T2 scan
        for MR, path in zip(MRs, MRs_paths):
            if "t2_" in MR['SeriesDescription'].value:
                MR_T2 = MR
                file_paths = path
            else:
                raise Exception

        # write files into new folder structure
        p = re.compile(r'VS-SEG-(\d+)')
        case_idx = int(p.findall(case)[0])
        old_T2_folder = os.path.dirname(file_paths)

        # Output path
        new_T2_path = os.path.join(output_path, 'vs_gk_' + str(case_idx) +'_T2.nii.gz')
        
        # Conversion DICOM to NIFTI using SITK
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(old_T2_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        sitk.WriteImage(image, new_T2_path)

if __name__ == "__main__":
    main()