import h5py
import os
import numpy as np
import torchvision.transforms as T

import argparse
from utils import get_device, load_config
from PIL import Image
from datasets_classes.dataset_h5 import Whole_Slide_Bag, WSI_Bag_Wrapper, Bag
from torchvision import transforms
import cv2


from munch import Munch


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

def split_macro_patch(sample_id, image_path, output_dir, out_patch_size, grayscale=False):
    
    if grayscale:
        macro_patch = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        macro_patch = cv2.imread(image_path)
        
    macro_patch_size = macro_patch.shape[0] # Assuming square macro patch !!!

    os.makedirs(output_dir, exist_ok=True)

    num_patches = macro_patch_size // out_patch_size

    for i in range(num_patches):
        for j in range(num_patches):
            patch = macro_patch[i * out_patch_size: (i + 1) * out_patch_size,
                                    j * out_patch_size: (j + 1) * out_patch_size]

            patch_filename = os.path.join(output_dir, f"{sample_id}_x_{i}_y_{j}.jpg")
            cv2.imwrite(patch_filename, patch)
            
def is_diffinfinite(config: Munch):
    
    input_dir = config.diffinfinite_macro_path
    output_base_dir = config.diffinfinite_out_path
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            output_dir = os.path.join(output_base_dir, os.path.splitext(filename)[0])  # Create separate output directory for each macro patch
            sample_id = os.path.splitext(filename)[0][6:]
            split_macro_patch(sample_id, image_path, output_dir, config.out_patch_size, config.grayscale_patches)


def is_wsi(config: Munch):

    output_path = config.wsi_output_path

    c_transforms = get_transforms(
        config.transforms.using_imagenet, config.transforms.prewhiten
    )

    bags_dataset = WSI_Bag_Wrapper()

    for slide_name in os.listdir(config.h5_source_path):
        if slide_name.endswith(config.hdf_extension):
            slide = Whole_Slide_Bag(
                config.h5_source_path + slide_name,
                target_patch_size=config.target_patch_size,
                pretrained=False,
                custom_transforms=c_transforms,
            )
            bags_dataset.add_bag(slide, slide_name.split(config.hdf_extension)[0])

    try:
        os.mkdir(output_path)
    except FileExistsError as e:
        print(f"\n\nOut Directory '{output_path}' already exists\n\n")
        exit(1)

    for bag in bags_dataset:
        cont = 0
        dir = os.path.join(output_path, bag.filename_noext)
        os.mkdir(dir)
        for img, coords in bag.wsi:
            if config.limit == -1 or cont < config.limit:
                tra = T.ToPILImage()
                im = tra(img.squeeze(0))
                patch_name = os.path.join(
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
