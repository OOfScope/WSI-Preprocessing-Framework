#!/bin/zsh
micromamba activate wsi-patch-extractor
exec=$(which python3)
exec .create_patches.py --source tcga-test --save_dir dest --no_auto_skip --patch_size 256 --preset tcga.csv