<div id="top" align="center">
  
# STDetph
**STDepth: Leveraging Semantic-Textural Information in Transformers for Self-supervised Monocular Depth Estimation**

<p align="center">
  <img src="assets/demo.gif" alt="example input output gif" width="450" />
</p>
STDepth (MPViT-xs 640x192)
</div>

## Description
This is the PyTorch implementation for STDepth. We build it based on the DDP version of Monodepth2, which have several new features:
* DDP training mode
* Cityscapes training and evaluation.
* Make3D evaluation


If you find our work useful in your research please consider citing our paper:

```
@article{gao2025stdepth,
  title={STDepth: Leveraging semantic-textural information in transformers for self-supervised monocular depth estimation},
  author={Gao, Xuanang and Wang, Bingchao and Ning, Zhiwei and Yang, Jie and Liu, Wei},
  journal={Computer Vision and Image Understanding},
  pages={104422},
  year={2025},
  publisher={Elsevier}
}
```



## Setup
Install the dependencies with:
```shell
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install scikit-image timm thop yacs opencv-python h5py six open3d mmsegmentation einops
```
We experiment with PyTorch 1.13.0, CUDA 11.7, Python 3.9. Other torch versions may also be okay.


## Preparing datasets
For KITTI dataset, you can prepare them as done in [Monodepth2](https://github.com/nianticlabs/monodepth2). Note that we directly train with the raw png images and do not convert them to jpgs. You also need to generate the groundtruth depth maps before training since the code will evaluate after each epoch. For the raw KITTI groundtruth (`eigen` eval split), run the following command. This will generate `gt_depths.npz` file in the folder `splits/kitti/eigen/`.
```shell
python export_gt_depth.py --data_path /home/datasets/kitti_raw_data --split eigen
```
Or if you want to use the improved KITTI groundtruth (`eigen_benchmark` eval split), please directly download it in this [link](https://www.dropbox.com/scl/fi/dg7eskv5ztgdyp4ippqoa/gt_depths.npz?rlkey=qb39aajkbhmnod71rm32136ry&dl=0). And then move the downloaded file (`gt_depths.npz`) to the folder `splits/kitti/eigen_benchmark/`.

For NYUv2 dataset, you can download the training and testing datasets as done in [StructDepth](https://github.com/SJTU-ViSYS/StructDepth).

For Make3D dataset, you can download it from [here](http://make3d.cs.cornell.edu/data.html#make3d).

For Cityscapes dataset, we follow the instructions in [ManyDepth](https://github.com/nianticlabs/manydepth). First Download `leftImg8bit_sequence_trainvaltest.zip` and `camera_trainvaltest.zip` in its [website](https://www.cityscapes-dataset.com/), and unzip them into the folder `/path/to/cityscapes`. Then preprocess CityScapes dataset using the followimg command:
```shell
python prepare_cityscapes.py \
--img_height 512 \
--img_width 1024 \
--dataset_dir .../cityscapes \
--dump_root .../cityscapes_preprocessed \
--seq_length 3 \
--num_threads 8
```
Remember to modify `--dataset_dir` and `--dump_root` to your own. The ground truth depth files are provided by ManyDepth in this [link](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip), which were converted from pixel disparities using intrinsics and the known baseline. Download this and unzip into `splits/cityscapes`


## Weights
You can download model weights in this [link](https://www.dropbox.com/scl/fo/0zeefm9e4kv0fzumqp490/h?rlkey=ev09rshvarnoyj9kr1qymkppl&dl=0), including three checkpoint files:
* Pretrained MPViT on ImageNet:   `mpvit_xsmall.pth` 
* Our final KITTI model (640×192) with an MPViT-xs backbone, trained on monocular (M) videos: `kitti_m_mpvit_640x192.pth`  
* Our final KITTI model (640×192) with an MPViT-xs backbone, trained on both monocular and stereo (MS) images: `kitti_ms_mpvit_640x192.pth`  
* Our final KITTI model (1024×320) with an MPViT-xs backbone, trained on monocular (M) videos: `kitti_m_mpvit_1024x320.pth`  
* Our final KITTI model (1024×320) with an MPViT-xs backbone, trained on both monocular and stereo (MS) images: `kitti_ms_mpvit_1024x320.pth`  


## Training
Before training, move the pretrained MPViT-xs weights, `mpvit-xs.pth`, to the folder `STDepth/ckpt/mpvit_xsmall.pth`.

```shell
cd /path/to/STDepth
mkdir ckpt
mv /path/to/mpvit_xsmall.pth ./ckpt
```

And you can see the training scripts in [run_kitti.sh](./run_kitti.sh), [run_nyu.sh](./run_nyu.sh) and [run_cityscapes.sh](./run_cityscapes.sh). Take the KITTI script as an example:
```shell
# CUDA_VISIBLE_DEVICES=0 python train.py \

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
--data_path .../kitti_raw_data \
--log_dir .../log_dir \
--model_name  model_name \
--width  640 \
--height 192 \
--batch_size 8 \
--num_epochs 40 \
--scale 0 \
--png \
--lr_nxt 1e-4 \
--learning_rate 5e-5 \
--scheduler_step_multi_size 15 30 \
--save_epoch 15 \
--save_step 300 \
--epoch_change 20 \
--frame_ids 0 \
```
Use `CUDA_VISIBLE_DEVICES=0 python train.py` to train with a single GPU. If you want to train with two or more GPUs, then use `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py` for DDP training.

Use `--data_path` flag to specify the dataset folder.

Use `--log_dir` flag to specify the logging folder.

Use `--exp_name` flag to specify the experiment name.

All output files (checkpoints, logs and tensorboard) will be saved in the directory `{log_dir}/{exp_name}`.

Use `--pretrained_path` flag to load a pretrained checkpoint if necessary.

Use `--split` flag to specify the training split on KITTI (see [Monodepth2](https://github.com/nianticlabs/monodepth2)), and default is eigen_zhou.

Look at [options.py](./options.py) to see the range of other training options.



## Evaluation
You can see the evaluation script in [evaluate.sh](./evaluate.sh). 

```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth_multi.py \
--pretrained_path ./weights_folder \
--backbone mpvit_xs \
--batch_size 8 \
--width 640 \
--height 192 \
--kitti_path .../kitti_raw_data \
--make3d_path .../make3d \
--cityscapes_path .../cityscapes \
--nyuv2_path .../nyu_v2 
# --post_process
```
This script will evaluate on KITTI (both raw and improved GT), NYUv2, Make3D and Cityscapes together. If you don't want to evaluate on some of these datasets, for example KITTI, just do not specify the corresponding `--kitti_path` flag. It will only evaluate on the datasets which you have specified a path flag.

If you want to evalute with post-processing, add the `--post_process` flag.


## Prediction

### Prediction for a single image
You can predict scaled disparity for a single image with:

```shell
python test_simple.py --image_path folder/test_image.jpg --pretrained_path ./weights_folder  --height 192 --width 640
```

The `--image_path` flag can also be a directory containing several images. In this setting, the script will predict all the images (use `--ext` to specify png or jpg) in the directory:

```shell
python test_simple.py --image_path folder --pretrained_path ./weights_folder  --height 192 --width 640 --ext png
```

### Prediction for a video

```shell
python test_video.py --image_path folder --pretrained_path ./weights_folder --height 192 --width 640 --ext png
```
Here the `--image_path` flag should be a directory containing several video frames. Note that these video frame files should be named in an ascending numerical order. For example, the first frame is named as `0000.png`, the second frame is named as `0001.png`, and etc. Then the script will output a GIF file.

## Acknowledgement
We have used codes from other wonderful open-source projects,
[SfMLearner](https://github.com/tinghuiz/SfMLearner/tree/master),
[Monodepth2](https://github.com/nianticlabs/monodepth2), [ManyDepth](https://github.com/nianticlabs/manydepth),[StructDepth](https://github.com/SJTU-ViSYS/StructDepth), [PlaneDepth](https://github.com/svip-lab/PlaneDepth) and [RA-Depth](https://github.com/hmhemu/RA-Depth). Thanks for their excellent works!
