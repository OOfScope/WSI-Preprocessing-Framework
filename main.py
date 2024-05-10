import h5py
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import pil_to_tensor

import argparse
from utils import get_device, load_config
from PIL import Image
from datasets_classes.dataset_h5 import Whole_Slide_Bag, WSI_Bag_Wrapper, Bag
from torchvision import transforms
import cv2
from glob import glob

from munch import Munch
import pandas as pd

from os import mkdir, makedirs, listdir, scandir
from os.path import basename, splitext, join
import torch
import matplotlib.pyplot as plt

from strategy import LabelingContext, TopKLabeling, TopKThrLabeling

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
    
    assert macro_patch_size > out_patch_size, 'unable to gen patches, check your config'
    assert macro_patch_size % out_patch_size == 0, 'unable to gen patches, check your config'

    
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
        
        
def cmap_mask(mask, diffclasses):
    if diffclasses == 5:
        color_map = {
            0: (0, 0, 0),      # Unknown
            1: (0, 0, 255),    # Carcinoma
            2: (255, 0, 0),    # Necrosis
            3: (0, 255, 0),    # Tumor Stroma
            4: (0, 255, 255),  # Others
        }
    elif diffclasses == 10:
            color_map = {
            0: (0, 0, 0),       # Black (Unknown)
            4: (255, 0, 0),     # Red (Carcinoma)
            7: (0, 255, 0),     # Green (Necrosis)
            8: (0, 0, 255),     # Blue (Tumor Stroma)
            9: (255, 255, 0),   # Yellow (Others)
            1: (255, 0, 255),   # Magenta (Alveole)
            3: (0, 255, 255),   # Cyan (Artifacts)
            5: (128, 128, 128), # Gray (Cartilage)
            6: (255, 128, 0),   # Orange (Connections)
            2: (0, 128, 255)    # Sky Blue (Artery)
        }
    
    unique_rgbs = set()

    for rgb in color_map.values():
        assert rgb not in unique_rgbs, "RGB values are not unique: error on" + str(rgb)
        unique_rgbs.add(rgb)
    
    cmapd_mask = np.zeros((mask.size[0], mask.size[1], 3), dtype=np.uint8)
    tmask = pil_to_tensor(mask)
    tmask = tmask.squeeze()
    
    for value, color in color_map.items():
        cmapd_mask[tmask == value] = color
    
    return cmapd_mask, transforms.ToPILImage()(cmapd_mask.squeeze())

def do_pre_split(image_tensors, mask_tensors, factor, out_path, gen_color_mapped_submasks, diffclasses):
    
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
                    
                    
                    if gen_color_mapped_submasks:
                        _, cmapped_mask = cmap_mask(sub_mask, diffclasses)
                        cmapped_mask.save(f'{out_path}/{counter:04d}/sub_cmapped_mask_{counter:04d}.png')

                    counter += 1
            
        except Exception as e:
            print(f"Error: {e}")
                
def calc_tensor_perc(tensor, diffinf_n_labels = 5) -> dict:
    unique_values, counts = np.unique(tensor, return_counts=True)
    perc_label_dict = {}

    for i in range(diffinf_n_labels):
        perc_label_dict[i] = 0
    
    total_elements = tensor.numel()
    for value, count in zip(unique_values, counts):
        perc = tensor[tensor == value].numel() / total_elements
        perc_label_dict[value] = perc
    
    assert sum(perc_label_dict.values()) == 1, 'sum of mask tensor percentages is not equal to 1'
    
    return perc_label_dict
            
def is_diffinfinite(config: Munch):
        
    input_dir = config.diffinfinite_macro_path
    output_base_dir = config.diffinfinite_out_path
    
    # masks are in png but I prefer to sort by substr and not by filext
    assets = glob(input_dir + '/**/*.jpg', recursive=True) +  glob(input_dir + '/**/*.png')
    
    assert len(assets) != 0, "No assets found, please check your input directory"

    image_tensors, mask_tensors = load_assets(assets)
    

    if config.split.enabled and config.patching_enabled:
        do_pre_split(image_tensors, mask_tensors, config.split.factor, config.diffinfinite_out_path + config.split.presplit_out_path, config.split.gen_color_mapped_submasks, config.diffclasses)
    

    if config.cmap_whole_masks:
        try:
            mkdir(output_base_dir + config.cmap_whole_masks_out)
        except FileExistsError as e:
            print(f"\n\nOut Directory '{output_base_dir + config.cmap_whole_masks_out}' already exists\n\n")
            

    for asset in assets:
        
        sample_id = splitext(basename(asset))[0]
        sample_id_number = sample_id[6:10]

        if config.cmap_whole_masks:
            
            if 'mask' in sample_id:
                whole_mask_pil = Image.open(asset)
                arr = np.array(whole_mask_pil, dtype='uint8')
                whole_mask_pil = Image.fromarray(arr)
                _, cmapped_whole_mask = cmap_mask(whole_mask_pil, config.diffclasses)
                cmapped_whole_mask.save(f'{output_base_dir}/{config.cmap_whole_masks_out}/cmapped_mask_{sample_id_number}.png')


        if 'mask' in sample_id and not config.using_masks:  # might remove it
            continue
        
        if config.patching_enabled:
            if config.split.enabled:
                continue
            else:    
                output_dir = join(output_base_dir, sample_id)  # Create separate output directory for each macro patch
                split_macro_patch(sample_id, sample_id_number, asset, output_dir, config.out_patch_size, config.grayscale_patches)

    if config.split.enabled and config.patching_enabled: # I want to patch my presplit dataset
        
        input_dir = config.diffinfinite_out_path + config.split.presplit_out_path
        output_dir = config.diffinfinite_out_path + config.split.patched_split_out
        assets = glob(input_dir + '/**/*.png', recursive=True)
        
        try:
            mkdir(output_dir)
        except Exception as e:
            print(e)
        
        for asset in assets:
            if 'cmapped' in asset:
                continue
            
            sample_id = splitext(basename(asset))[0]
            
            if 'mask' in asset:
                sample_id_number = sample_id[9:14]
                asset_out_dir = output_dir + '/masks/' + sample_id_number                
                # alternative tree:
                #asset_out_dir = output_dir + sample_id_number + /masks/
            else:
                sample_id_number = sample_id[10:14]
                asset_out_dir = output_dir + '/patches/' + sample_id_number
                # alternative tree:
                #asset_out_dir = output_dir + sample_id_number + /patches/
                
            split_macro_patch(sample_id, sample_id_number, asset, asset_out_dir, config.out_patch_size, config.grayscale_patches)
            
            
            
            

    if config.annotator.enabled:
        
        supported_strategies = ['TopKLabeling', 'TopKThrLabeling']
                            
        assert config.annotator.strategy in supported_strategies, 'unsupported annotator strategy'
        assert config.annotator.k > 0, 'k must be greater than 0'
        assert config.annotator.thr > 0.001, 'thr must be greater than 0.001'
        
        labeling_context = None
        
        if config.annotator.strategy == 'TopKLabeling':
            labeling_context = LabelingContext(TopKLabeling(config.annotator.k))
        elif config.annotator.strategy == 'TopKThrLabeling':
            labeling_context = LabelingContext(TopKThrLabeling(config.annotator.k, config.annotator.thr))

        
        
        if not config.split.enabled and config.patching_enabled:
            print(f"Sorry this leaf is not implemented at the moment")
            exit(1)
        
        assert config.annotator.strategy in supported_strategies, 'unsupported annotator strategy'
        
        
        if config.split.enabled and config.patching_enabled:
            
            dic = {}
            masks_root_input_dir = config.diffinfinite_out_path + config.split.presplit_out_path
            
            samples_paths = [ f.path for f in scandir(masks_root_input_dir) if f.is_dir() ]
            samples_paths.sort()
            print(f'\nFound {samples_paths} presplits')
            
            assert len(samples_paths) > 0, 'invalid presplits found'
            assert len(samples_paths) % 2 == 0, 'potential invalid presplits found'
            
            if config.diffclasses != 5:
                print(f'\n\n!! Warning: diffclasses is not 5, halting here\n\n')
                exit(0)
            
            classes = ['Unknown', 'Carcinoma', 'Necrosis', 'Tumor_Stroma', 'Others']
            ABS_classes = [f'ABS_{c}' for c in classes]
            PERC_classes = [f'PERC_{c}' for c in classes]
            
            
            dic['sample_id'] = []
            
            for c in classes:
                dic[c] = []
                
            for c in ABS_classes:
                dic[c] = []
                
            for c in PERC_classes:
                dic[c] = []
                
                
            strategy_signature = labeling_context.labeling_strategy.get_strategy_signature()
            dic['strategy_signature'] = [strategy_signature] * len(samples_paths)
            
            for sample_path in samples_paths:
                for asset in listdir(sample_path):
                    # alternative: regexpression
                    if 'mask' not in asset:
                        continue
                    if 'cmapped' in asset:
                        continue
                    
                    sample_id_number = asset[9:13]
                    maskdir = sample_path + '/' + asset
                    mask = Image.open(maskdir)
                    mask_tensor = pil_to_tensor(mask)
                    abs_labels = np.unique(mask_tensor)
                    print(f'\nsid "{asset}" shows absolute labels: {abs_labels}')
                    
                    if config.diffclasses != 5:
                        continue
                    
                    
                    dic['sample_id'].append(sample_id_number)

                    for c in ABS_classes:
                        dic[c].append(0)
                    
                    for abs_l in abs_labels:
                        dic[ABS_classes[abs_l]][-1] = 1
                    
                    
                    percs = calc_tensor_perc(mask_tensor)
                    labeling_context.set_perc_dict(percs)

                    for prc_l in percs.keys():
                        dic[PERC_classes[prc_l]].append(percs[prc_l])                            


                    processed_labels_with_percs = labeling_context.process_annotation()
                    processed_labels = [label for label, perc in processed_labels_with_percs]

                    print(f"{sample_id_number} processed labels: {processed_labels}\n")
                    
                    
                    for c in classes:
                        dic[c].append(0)
                        
                    for l in processed_labels:
                        dic[classes[l]][-1] = 1


            df = pd.DataFrame(dic)
            df.set_index('sample_id', inplace=True)
            df.to_csv(config.annotator.csv_filename)

        if config.annotator.plot_class_distribution or config.annotator.balancer.enabled:
            plot_df = df[classes]
            column_sums = plot_df.sum()

        if config.annotator.plot_class_distribution:
            plot_out_dir = 'plots/'
            try:
                mkdir(plot_out_dir)
            except Exception as e:
                print(e)
            plot_df.plot(kind='bar', stacked=True)
            plt.savefig(plot_out_dir + 'labels_per_sample.png')
            
            
            plt.figure(figsize=(8, 6))
            column_sums.plot(kind='bar', color='skyblue')
            plt.title('Total processed labels in the dataset')
            plt.xlabel('Columns')
            plt.ylabel('Samples having the label')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(plot_out_dir + 'classes_across_samples.png')
            
        if config.annotator.balancer.enabled:
            if any(column_sums == 0):
                print('\n-----------------------------------')
                print('!! Some classes are not present at all in the dataset, balancer can not be applied')
                print(f"Classes with 0 samples: {column_sums[column_sums == 0].index.tolist()}")
                print('-----------------------------------\n')                
                return
            
            overrepresented_class = column_sums.idxmax()
            
            

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
        description="WSI-Preprocessing-Framework: synthetic WSI preprocessing end to end."
    )

    parser.add_argument(
        "--config_path",
        default="config.yaml",
        type=str,
        help="wsi-preprocessing-framework config filepath",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    if config.is_diffinfinite:
        is_diffinfinite(config)
    else:
        is_wsi(config)


if __name__ == "__main__":
    main()
