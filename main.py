import h5py
import numpy as np
import torchvision.transforms as T

import argparse
from utils import get_device, load_config
from PIL import Image
from datasets_classes.dataset_h5 import Whole_Slide_Bag, WSI_Bag_Wrapper, Bag
from torchvision import transforms
import cv2
from glob import glob

from munch import Munch
import pandas as pd

from os import mkdir, makedirs, listdir
from os.path import basename, splitext, join
import torch

def get_transforms(using_imagenet=False, prewhiten=False) -> T.Compose:
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if using_imagenet:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    if prewhiten:
        t = [T.ToTensor(), T.Normalize(mean, std)]
    else:
        t = [T.ToTensor()]

    return T.Compose(t)

def split_macro_patch(sample_id, sample_id_number, image_path, output_dir, out_patch_size, grayscale=False):
    
    if grayscale:
        macro_patch = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        macro_patch = cv2.imread(image_path)
        
    macro_patch_size = macro_patch.shape[0] # Assuming square macro patch !!!
    
    makedirs(output_dir, exist_ok=True)

    num_patches = macro_patch_size // out_patch_size

    for i in range(num_patches):
        for j in range(num_patches):
            patch = macro_patch[i * out_patch_size: (i + 1) * out_patch_size,
                                    j * out_patch_size: (j + 1) * out_patch_size]
            patch_filename = join(output_dir, f"{sample_id_number}_x_{i}_y_{j}.jpg")
            cv2.imwrite(patch_filename, patch)
            
def load_assets(asset_paths):
    try:
        # sorting according to basename and not full path (bug: "mask" is a folder hence wrong classification)
        image_paths = [path for path in asset_paths if not basename(path).endswith('_mask.png')]
        mask_paths = [path for path in asset_paths if basename(path).endswith('_mask.png')]
        
        image_paths.sort(key=lambda x: int(basename(x).split('.')[0][-4:]))
        mask_paths.sort(key=lambda x: int(basename(x).split('_')[0][-4:]))
        
        images = []
        masks = []
        
        for image_path, mask_path in zip(image_paths, mask_paths):
            image = Image.open(image_path)
            images.append(transforms.ToTensor()(image))
            
            mask = Image.open(mask_path).convert('L')  # Convert to grayscale
            masks.append(transforms.ToTensor()(mask))
        
        # lists to tensors
        images_tensor = torch.stack(images)
        masks_tensor = torch.stack(masks)
        
        assert len(images) != 0 and len(masks) != 0, "images or masks list is empty, please check"
        assert images_tensor.shape[0] == masks_tensor.shape[0], "Shape mismatch, some images or some masks might be missing, plase check"
        
        return images_tensor, masks_tensor
    
    except Exception as e:
        print(f"Error: {e}")       
            
def do_pre_split(image_tensors, mask_tensors, factor, out_path):
    
    try:
        mkdir(out_path)
    except Exception as e:
        if(len(listdir(out_path)) != 0):
            print(e)
    
    counter = 0
    
    for img, mask in zip(image_tensors, mask_tensors):
        try:
            
            channels, height, width = img.size()

            sub_width = width // factor
            sub_height = height // factor
            
            for i in range(factor):
                for j in range(factor):
                    left = j * sub_width
                    upper = i * sub_height
                    right = (j + 1) * sub_width
                    lower = (i + 1) * sub_height

                    sub_image_tensor = img[:, upper:lower, left:right]
                    sub_mask_tensor = mask[:, upper:lower, left:right]                    
                    
                    sub_image = transforms.ToPILImage()(sub_image_tensor.squeeze())
                    sub_mask = transforms.ToPILImage()(sub_mask_tensor.squeeze())
                    
                    try:
                        mkdir(f'{out_path}/{counter:04d}')
                    except Exception as e:
                        print(e)
                        
                    sub_image.save(f'{out_path}/{counter:04d}/sub_image_{counter:04d}.png')
                    sub_mask.save(f'{out_path}/{counter:04d}/sub_mask_{counter:04d}.png')
                    counter += 1
            
        except Exception as e:
            print(f"Error: {e}")

                
            
def is_diffinfinite(config: Munch):
        
    input_dir = config.diffinfinite_macro_path
    output_base_dir = config.diffinfinite_out_path
    
    # masks are in png but I prefer to sort by substr and not by filext
    assets = glob(input_dir + '/**/*.jpg', recursive=True) +  glob(input_dir + '/**/*.png')
    
    image_tensors, mask_tensors = load_assets(assets)
    
    
    if config.split.enabled:
        do_pre_split(image_tensors, mask_tensors, config.split.factor, config.split.presplit_out_path)
    
    
    df = pd.DataFrame()



    for image_path in assets:
        sample_id = splitext(basename(image_path))[0]

        if 'mask' in sample_id and not config.using_masks:
            continue
        
        sample_id_number = sample_id[6:10]
        
        output_dir = join(output_base_dir, sample_id)  # Create separate output directory for each macro patch
        split_macro_patch(sample_id, sample_id_number, image_path, output_dir, config.out_patch_size, config.grayscale_patches)


def is_wsi(config: Munch):

    output_path = config.wsi_output_path

    c_transforms = get_transforms(
        config.transforms.using_imagenet, config.transforms.prewhiten
    )

    bags_dataset = WSI_Bag_Wrapper()

    for slide_name in listdir(config.h5_source_path):
        if slide_name.endswith(config.hdf_extension):
            slide = Whole_Slide_Bag(
                config.h5_source_path + slide_name,
                target_patch_size=config.target_patch_size,
                pretrained=False,
                custom_transforms=c_transforms,
            )
            bags_dataset.add_bag(slide, slide_name.split(config.hdf_extension)[0])

    try:
        mkdir(output_path)
    except FileExistsError as e:
        print(f"\n\nOut Directory '{output_path}' already exists\n\n")
        exit(1)

    for bag in bags_dataset:
        cont = 0
        dir = join(output_path, bag.filename_noext)
        mkdir(dir)
        for img, coords in bag.wsi:
            if config.limit == -1 or cont < config.limit:
                tra = T.ToPILImage()
                im = tra(img.squeeze(0))
                patch_name = join(
                    dir,
                    "_x_" + str(int(coords[0])) + "_y_" + str(int(coords[1])) + ".jpg",
                )
                im.save(patch_name)
                cont += 1
            else:
                break
        print(f"{bag.filename_noext} done!")


def main():

    parser = argparse.ArgumentParser(
        description="WSI-Patch-Extractor: Extract patches as images from WSI files."
    )

    parser.add_argument(
        "--config_path",
        default="config.yaml",
        type=str,
        help="wsi-patch-extractor config filepath",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    if config.is_diffinfinite:
        is_diffinfinite(config)
    else:
        is_wsi(config)


if __name__ == "__main__":
    main()
