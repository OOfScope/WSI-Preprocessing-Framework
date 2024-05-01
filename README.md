# WSI Preprocessing and Patching


## Synthetic dataset pipeline (diffinfinite)

### Initial setup

Let's assume that we start from a synthetic parent dataset made up of WSI `4096x4096` with their respective masks with values going from 0 to (n_classes - 1), recall the [diffinfinite](https://arxiv.org/abs/2306.13384) paper which reports that we can choose a generation on 5 classes or on 10 classes


> example starting synth wsi (ω=3, 4096x4096)
<img src="res/init_sample.jpg" width="250px">

### Dataset pre-split extraction

To obtain a dataset with wsi of comparable size to other datasets present in the literature such as Camelyon16, we do not compute yet the labels of the masks but rather perform a first split (defined as pre-split) of each wsi and its mask.

The pre-split 'denominator' can be configured through the `split.factor` parameter in the `config.yml` file, therefore with a `factor=4` on the starting size previously assumed we will therefore obtain from each parent wsi and parent mask as many as 16 wsi with their respective masks all of size `1024x1024`

You can enable pre-split by turning `split.enabled` to `True`

> Remember: labels will be computed at this mask resolution level

> example pre-split synth wsi (ω=3, 1024x1024, coords=(0,0))
<img src="res/presplit_sample.png" width="250px">


#### Advantages

This approach allows for more time-efficient and robust generation on the algorithmic and operator side, avoiding generating dozens of smaller wsi but instead augmenting the dataset in post.
It is therefore possible to increase the size of the parent wsi as long as the system's VRAM allows it

### Whole Masks and Sub Masks color mapping

For better data understanding is possible to enable the `cmap_whole_masks`and/or the `gen_color_mapped_submasks` parameters in `config.yml`

In this way we can bring a mask (visually all black but in reality contains values very close to black from 0 to 4) into more recognizable color codes

> whole mask color mapping:

<img src="res/cmapped_whole_mask.png" width="250px">


> sub mask color mapping 
> 1/16 of the above whole mask, coordinates (0,0):

<img src="res/sub_cmapped_mask.png" width="250px">


#### Color Codes

    Unknown: rgb(0, 0, 0)   
    Carcinoma: rgb(0, 0, 255)
    Necrosis: rgb(255, 0, 0)
    Tumor Stroma: rgb(0, 255, 0)
    Others: rgb(0, 255, 255)

### Patching

Now we come to the actual patching, if we had enabled the pre-split we will patch the out of the pre-split otherwise the WSI parent images


## TCGA Dataset

# Credits

The synthetic dataset has been obtained through [diffinfinite](https://arxiv.org/abs/2306.13384) DM

The TCGA patching work is based on [CLAM](https://github.com/mahmoodlab/CLAM) slow patching algorithm