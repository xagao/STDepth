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