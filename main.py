
import h5py
import os
import numpy as np
import torchvision.transforms as T

import argparse
from utils import get_device, load_config
from PIL import Image
from datasets_classes.dataset_h5 import Whole_Slide_Bag, WSI_Bag_Wrapper, Bag
from torchvision import transforms

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


def main():
    
    parser = argparse.ArgumentParser(description='WSI-Patch-Extractor: Extract patches as images from WSI files.')

    parser.add_argument('--config_path', default='config.yaml', type=str, help='deepmiml-fw config filepath')
    args = parser.parse_args()

    config = load_config(args.config_path)

    src_path = '/Users/mich/Desktop/WSI-Patch-Extractor/new_dest_slow/patches/LUAD-TCGA-05-4417-01Z-00-DX1.h5'
    slide_ext = '.svs'
    bag_candidate_idx = 0



    csvpath = os.path.join(src_path, "process_list_autogen.csv")
    #output_path = os.path.join(src_path, "img_patches")

    output_path = 'out/'

    c_transforms = get_transforms(config.transforms.using_imagenet, config.transforms.prewhiten)

    bags_dataset = WSI_Bag_Wrapper()

    for slide_name in os.listdir(config.h5_source_path):
        if slide_name.endswith(config.hdf_extension):
            slide = Whole_Slide_Bag(config.h5_source_path + slide_name, target_patch_size=config.target_patch_size, pretrained=False, custom_transforms=c_transforms)
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
            if (config.limit == -1 or cont < config.limit):
                tra = T.ToPILImage()
                im = tra(img.squeeze(0))
                patch_name = os.path.join(dir, "_x_"+str(int(coords[0]))+"_y_"+str(int(coords[1]))+".jpg")
                im.save(patch_name)
                cont += 1
            else:
                break
        print(f"{bag.filename_noext} done!")




if __name__ == "__main__":
    main()