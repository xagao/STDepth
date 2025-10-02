# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import logging
import torch.distributed as dist

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
from thop import profile, clever_format

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        if self.opt.global_rank == 0:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.save_opts()
            if not os.path.exists(os.path.join(self.log_path, 'ckpt.pth')):
                setup_logging(os.path.join(self.log_path, 'logger.log'), rank=self.opt.global_rank)
                logging.info("Experiment is named: %s", self.opt.model_name)
                logging.info("Saving to: %s", os.path.abspath(self.log_path))
                logging.info("GPU numbers: %d", self.opt.world_size)
                logging.info("Training dataset: %s", self.opt.dataset)
            else:
                setup_logging(os.path.join(self.log_path, 'logger.log'), filemode='a', rank=self.opt.global_rank)

            self.writers = {}
            for mode in ["train"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, "tensorboard", mode))
        if self.opt.world_size > 1:
            dist.barrier()

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.batch_start = 0
        
        if self.opt.world_size > 1:
            dist.barrier()
        self.device = torch.device('cuda', self.opt.local_rank)

        self.num_gpus = torch.cuda.device_count()
        assert self.num_gpus > 0, "No GPU found or CUDA is not enabled."

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.mpvit.mpvit_xsmall()
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
    
        self.num_ch_enc = np.array([64, 128, 192, 256, 256])
        
        self.models["depth"] = networks.DepthDecoder_Mix(
            self.num_ch_enc, range(4))
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        # self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        if self.opt.world_size > 1:
            for k in self.models.keys():
                self.models[k] = nn.SyncBatchNorm.convert_sync_batchnorm(
                self.models[k])
                self.models[k] = nn.parallel.DistributedDataParallel(self.models[k], device_ids=[self.opt.local_rank], 
                                                                     output_device=self.opt.local_rank, find_unused_parameters=True)

        
        print("Learning rate scheduler info:")
        print(" - Base learning rate:", self.model_lr_scheduler.base_lrs)
        # print(" - Milestones:", self.model_lr_scheduler.milestones)
        # print(" - Gamma (decay factor):", self.model_lr_scheduler.gamma)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        
        num_train_samples = len(train_filenames)
        self.num_steps_per_epoch = num_train_samples // self.opt.world_size // self.opt.batch_size
        if self.opt.save_step != 0:
            self.save_step = self.opt.save_step
        else:
            self.save_step = self.num_steps_per_epoch
        self.num_total_steps = self.num_steps_per_epoch * self.opt.num_epochs

        if self.opt.dataset == "cityscapes":
            train_dataset = self.dataset(
                self.opt.data_path_pre, train_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, self.opt.num_scales, is_train=True, img_ext=img_ext)            
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        if self.opt.world_size > 1:
            #self.sampler = datasets.CustomDistributedSampler(train_dataset, self.opt.seed)
            self.sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.train_batch_sampler = torch.utils.data.BatchSampler(self.sampler, self.opt.batch_size, drop_last=True)
        else:
            self.sampler = datasets.CustomSampler(train_dataset, self.opt.seed)
        
        if self.opt.world_size > 1:
            self.train_loader = DataLoader(train_dataset,
                batch_sampler=self.train_batch_sampler, pin_memory=True, num_workers=self.opt.num_workers)
        else:
            # self.train_loader = DataLoader(
            # train_dataset, self.opt.batch_size, shuffle=False, sampler=self.sampler, num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)


        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.detail_guide = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.detail_guide[scale] = DetailGuide(self.opt.batch_size, h, w)
            self.detail_guide[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        if self.opt.world_size > 1:
            dist.barrier()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        #torch.autograd.set_detect_anomaly(True)
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

            if self.opt.global_rank == 0:
                if (self.epoch + 1) % self.opt.save_frequency == 0:
                    self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # self.model_lr_scheduler.step()
        logging.info("Training epoch {}\n".format(self.epoch))

        print("Training")
        self.sampler.set_epoch(self.epoch)
        self.set_train()

        if self.opt.world_size > 1:
            dist.barrier()

        if self.epoch == self.opt.epoch_change:
            lr = self.opt.lr_nxt
            for param_group in self.model_optimizer.param_groups:
                param_group['lr'] = lr

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0
            # save_phase = self.step % self.opt.save_step == 0 and self.step > 0
            save_phase = self.step % self.save_step == 0 and self.step > self.num_steps_per_epoch * self.opt.save_epoch

            if early_phase or late_phase:
                if self.opt.world_size > 1:
                    dist.barrier()
                    for k in losses.keys():
                        dist.all_reduce(losses[k], op=dist.ReduceOp.SUM)
                        losses[k] = losses[k] / self.opt.world_size
                    dist.barrier()
                self.log_time(batch_idx, duration, losses["loss_HiS"].cpu().data, losses["loss_MiS"].cpu().data, losses["loss_LoS"].cpu().data, losses["loss_consis_HiS"].cpu().data, losses["loss_consis_LoS"].cpu().data, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
            
            if self.opt.global_rank == 0:
                if save_phase and self.opt.save_step != 0:
                    self.save_model()

                    self.log("train", inputs, outputs, losses)
                    # self.val()

            self.step += 1

        self.model_lr_scheduler.step()

    
    def profile_once(self, encoder, decoder, x):
        x_e = x[0, :, :, :].unsqueeze(0)
        x_d = encoder(x_e)
        # flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
        flops_e, params_e = 0, 0
        flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

        flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
        flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
        flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

        print(flops_d, params)

        return flops, params, flops_e, params_e, flops_d, params_d

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        if self.opt.pose_model_type == "shared":
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            if self.epoch >= self.opt.epoch_change:
                features_HiS = self.models["encoder"](inputs["color_HiS", 0, 0])
                outputs["out_HiS"] = self.models["depth"](features_HiS)
                features_MiS = self.models["encoder"](inputs["color_MiS_aug", 0, 0])
                outputs["out_MiS"] = self.models["depth"](features_MiS)
                features_LoS = self.models["encoder"](inputs["color_LoS", 0, 0])
                outputs["out_LoS"] = self.models["depth"](features_LoS)
            
            
            else:
                
                features_MiS = self.models["encoder"](inputs["color_MiS_aug", 0, 0])
                outputs["out_MiS"] = self.models["depth"](features_MiS)
                disp_MiS = outputs["out_MiS"][("disp", 0)]

                x_starts = torch.round(self.opt.width * inputs[("dxy_HiS")][:, 0] / inputs[("resize_HiS")][:, 0]).int()
                y_starts = torch.round(self.opt.height * inputs[("dxy_HiS")][:, 1] / inputs[("resize_HiS")][:, 1]).int()
                width_unions = torch.round(self.opt.width * self.opt.width / inputs[("resize_HiS")][:, 0]).int()
                height_unions = torch.round(self.opt.height * self.opt.height / inputs[("resize_HiS")][:, 1]).int()

                disp_HiS = []
                for i in range(self.opt.batch_size):
                    disp_H = disp_MiS[i, :, y_starts[i]:y_starts[i]+height_unions[i], x_starts[i]:x_starts[i]+width_unions[i]].unsqueeze(0)
                    disp_H = F.interpolate(disp_H, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    disp_HiS.append(disp_H)

                outputs["out_HiS"] = torch.cat(disp_HiS, dim=0)

                disp_LoS = []
                mask_LoS = []
                for i in range(self.opt.batch_size):

                    width_un_LoS = int(inputs[("resize_LoS")][i, 0])
                    height_un_LoS = int(inputs[("resize_LoS")][i, 1])
                    disp_L_resize = F.interpolate(disp_MiS[i].unsqueeze(0), [height_un_LoS, width_un_LoS], mode="bilinear", align_corners=False)

                    disp_L = torch.ones_like(disp_MiS[i].unsqueeze(0))
                    disp_L[:, :, :height_un_LoS, :width_un_LoS] = disp_L_resize

                    mask_L = torch.ones_like(disp_L)
                    mask_L[:, :, height_un_LoS:, :] = 0
                    mask_L[:, :, :, width_un_LoS:] = 0

                    disp_LoS.append(disp_L)
                    mask_LoS.append(mask_L)

                outputs["out_LoS"] = torch.cat(disp_LoS, dim=0)
                outputs["out_LoS_mask"] = torch.cat(mask_LoS, dim=0)


        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features_MiS)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features_MiS))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_MiS", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_pose_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            if self.epoch >= self.opt.epoch_change:
                disp_HiS = outputs["out_HiS"][("disp", scale)]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    
                    source_scale = 0
                _, depth_HiS = disp_to_depth(disp_HiS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_HiS", 0, scale)] = depth_HiS


                disp_MiS = outputs["out_MiS"][("disp", scale)]
                _, depth_MiS = disp_to_depth(disp_MiS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_MiS", 0, scale)] = depth_MiS


                disp_LoS = outputs["out_LoS"][("disp", scale)]
                _, depth_LoS = disp_to_depth(disp_LoS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_LoS", 0, scale)] = depth_LoS
            
            else:
                disp_HiS = outputs["out_HiS"]
                if self.opt.v1_multiscale:
                    source_scale = scale
                else:
                    source_scale = 0
                _, depth_HiS = disp_to_depth(disp_HiS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_HiS", 0, scale)] = depth_HiS


                disp_MiS = outputs["out_MiS"][("disp", scale)]
                _, depth_MiS = disp_to_depth(disp_MiS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_MiS", 0, scale)] = depth_MiS


                disp_LoS = outputs["out_LoS"]
                _, depth_LoS = disp_to_depth(disp_LoS, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth_LoS", 0, scale)] = depth_LoS


            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points_HiS = self.backproject_depth[source_scale](
                    depth_HiS, inputs[("inv_K_HiS", source_scale)], inputs[("dxy_HiS")])
                pix_coords_HiS = self.project_3d[source_scale](
                    cam_points_HiS, inputs[("K_HiS", source_scale)], T, inputs[("dxy_HiS")])
                outputs[("sample_HiS", frame_id, scale)] = pix_coords_HiS
                outputs[("color_HiS", frame_id, scale)] = F.grid_sample(
                    inputs[("color_HiS", frame_id, source_scale)],
                    outputs[("sample_HiS", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)

                cam_points_MiS = self.backproject_depth[source_scale](
                    depth_MiS, inputs[("inv_K_MiS", source_scale)], inputs[("dxy_MiS")])
                pix_coords_MiS = self.project_3d[source_scale](
                    cam_points_MiS, inputs[("K_MiS", source_scale)], T, inputs[("dxy_MiS")])
                outputs[("sample_MiS", frame_id, scale)] = pix_coords_MiS
                outputs[("color_MiS", frame_id, scale)] = F.grid_sample(
                    inputs[("color_MiS", frame_id, source_scale)],
                    outputs[("sample_MiS", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)

                cam_points_LoS = self.backproject_depth[source_scale](
                    depth_LoS, inputs[("inv_K_LoS", source_scale)], inputs[("dxy_LoS")])
                pix_coords_LoS = self.project_3d[source_scale](
                    cam_points_LoS, inputs[("K_LoS", source_scale)], T, inputs[("dxy_LoS")])
                outputs[("sample_LoS", frame_id, scale)] = pix_coords_LoS
                outputs[("color_LoS", frame_id, scale)] = F.grid_sample(
                    inputs[("color_LoS", frame_id, source_scale)],
                    outputs[("sample_LoS", frame_id, scale)],
                    padding_mode="border",
                    align_corners = True)


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_reprojection_loss_disp(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.5 * ssim_loss + 0.5 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        loss_HiS = 0
        loss_MiS = 0
        loss_LoS = 0
        device = torch.device("cuda")

        loss_consis_HiS = 0
        loss_consis_LoS = 0

        for scale in self.opt.scales:
            reprojection_losses_HiS = []
            reprojection_losses_MiS = []
            reprojection_losses_LoS = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0
            
            """change loss function"""
            if self.epoch >= self.opt.epoch_change:
                disp_HiS = outputs["out_HiS"][("disp", scale)]
                color_HiS = inputs[("color_HiS", 0, source_scale)]
                target_HiS = inputs[("color_HiS", 0, source_scale)]

                disp_MiS = outputs["out_MiS"][("disp", scale)]
                color_MiS = inputs[("color_MiS", 0, source_scale)]
                target_MiS = inputs[("color_MiS", 0, source_scale)]

                disp_LoS = outputs["out_LoS"][("disp", scale)]
                color_LoS = inputs[("color_LoS", 0, source_scale)]
                target_LoS = inputs[("color_LoS", 0, source_scale)]
            
            else:
                disp_HiS = outputs["out_HiS"]
                color_HiS = inputs[("color_HiS", 0, source_scale)]
                target_HiS = inputs[("color_HiS", 0, source_scale)]

                disp_MiS = outputs["out_MiS"][("disp", scale)]
                color_MiS = inputs[("color_MiS", 0, source_scale)]
                target_MiS = inputs[("color_MiS", 0, source_scale)]

                disp_LoS = outputs["out_LoS"]
                color_LoS = inputs[("color_LoS", 0, source_scale)]
                target_LoS = inputs[("color_LoS", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred_HiS = outputs[("color_HiS", frame_id, scale)]
                reprojection_losses_HiS.append(self.compute_reprojection_loss(pred_HiS, target_HiS))
                pred_MiS = outputs[("color_MiS", frame_id, scale)]
                reprojection_losses_MiS.append(self.compute_reprojection_loss(pred_MiS, target_MiS))
                pred_LoS = outputs[("color_LoS", frame_id, scale)]
                reprojection_losses_LoS.append(self.compute_reprojection_loss(pred_LoS, target_LoS))

            reprojection_losses_HiS = torch.cat(reprojection_losses_HiS, 1)
            reprojection_losses_MiS = torch.cat(reprojection_losses_MiS, 1)
            reprojection_losses_LoS = torch.cat(reprojection_losses_LoS, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses_HiS = []
                identity_reprojection_losses_MiS = []
                identity_reprojection_losses_LoS = []
                for frame_id in self.opt.frame_ids[1:]:
                    # if frame_id == 's':
                    pred_HiS = inputs[("color_HiS", frame_id, source_scale)]
                    identity_reprojection_losses_HiS.append(
                        self.compute_reprojection_loss(pred_HiS, target_HiS))

                    pred_MiS = inputs[("color_MiS", frame_id, source_scale)]
                    identity_reprojection_losses_MiS.append(
                        self.compute_reprojection_loss(pred_MiS, target_MiS))

                    pred_LoS = inputs[("color_LoS", frame_id, source_scale)]
                    identity_reprojection_losses_LoS.append(
                        self.compute_reprojection_loss(pred_LoS, target_LoS))

                identity_reprojection_losses_HiS = torch.cat(identity_reprojection_losses_HiS, 1)
                identity_reprojection_losses_MiS = torch.cat(identity_reprojection_losses_MiS, 1)
                identity_reprojection_losses_LoS = torch.cat(identity_reprojection_losses_LoS, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss_HiS = identity_reprojection_losses_HiS.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss_HiS = identity_reprojection_losses_HiS
                    identity_reprojection_loss_MiS = identity_reprojection_losses_MiS
                    identity_reprojection_loss_LoS = identity_reprojection_losses_LoS

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses_HiS *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss_HiS = reprojection_losses_HiS.mean(1, keepdim=True)
            else:
                reprojection_loss_HiS = reprojection_losses_HiS
                reprojection_loss_MiS = reprojection_losses_MiS
                reprojection_loss_LoS = reprojection_losses_LoS

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss_HiS += torch.randn(
                    identity_reprojection_loss_HiS.shape).cuda() * 0.00001
                identity_reprojection_loss_MiS += torch.randn(
                    identity_reprojection_loss_MiS.shape).cuda() * 0.00001
                identity_reprojection_loss_LoS += torch.randn(
                    identity_reprojection_loss_LoS.shape).cuda() * 0.00001

                combined_HiS = torch.cat((identity_reprojection_loss_HiS, reprojection_loss_HiS), dim=1)
                combined_MiS = torch.cat((identity_reprojection_loss_MiS, reprojection_loss_MiS), dim=1)
                combined_LoS = torch.cat((identity_reprojection_loss_LoS, reprojection_loss_LoS), dim=1)
            else:
                combined_HiS = reprojection_loss_HiS

            if combined_HiS.shape[1] == 1:
                to_optimise_HiS = combined_HiS
            else:
                to_optimise_HiS, idxs_HiS = torch.min(combined_HiS, dim=1)
                to_optimise_MiS, idxs_MiS = torch.min(combined_MiS, dim=1)
                to_optimise_LoS, idxs_LoS = torch.min(combined_LoS, dim=1)

                if not self.epoch >= self.opt.epoch_change:
                    to_optimise_LoS = to_optimise_LoS * outputs["out_LoS_mask"]

            if not self.opt.disable_automasking:
                outputs["identity_selection_His/{}".format(scale)] = (
                    idxs_HiS > identity_reprojection_loss_HiS.shape[1] - 1).float()
                outputs["identity_selection_Mis/{}".format(scale)] = (
                    idxs_MiS > identity_reprojection_loss_HiS.shape[1] - 1).float()
                outputs["identity_selection_Los/{}".format(scale)] = (
                    idxs_LoS > identity_reprojection_loss_HiS.shape[1] - 1).float()

            mask4consis = torch.ones(self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
            mask4consis[idxs_MiS == 0] = 0 #[12,192,640]
            # mask4consis[idxs_MiS == 1] = 0
            mask4consis = mask4consis.unsqueeze(1).detach() #[12,1,192,640]

            loss_HiS += to_optimise_HiS.mean()
            mean_disp_HiS = disp_HiS.mean(2, True).mean(3, True)
            norm_disp_HiS = disp_HiS / (mean_disp_HiS + 1e-7)
            smooth_loss_HiS = get_smooth_loss(norm_disp_HiS, color_HiS)
            loss_HiS += self.opt.disparity_smoothness * smooth_loss_HiS

            loss_MiS += to_optimise_MiS.mean()
            mean_disp_MiS = disp_MiS.mean(2, True).mean(3, True)
            norm_disp_MiS = disp_MiS / (mean_disp_MiS + 1e-7)
            smooth_loss_MiS = get_smooth_loss(norm_disp_MiS, color_MiS)
            loss_MiS += self.opt.disparity_smoothness * smooth_loss_MiS

            loss_LoS += to_optimise_LoS.mean()
            mean_disp_LoS = disp_LoS.mean(2, True).mean(3, True)
            norm_disp_LoS = disp_LoS / (mean_disp_LoS + 1e-7)
            smooth_loss_LoS = get_smooth_loss(norm_disp_LoS, color_LoS)
            loss_LoS += self.opt.disparity_smoothness * smooth_loss_LoS
            
            if self.epoch >= self.opt.epoch_change:
                disp_diff_batch = 0
                for i in range(self.opt.batch_size):
                    x_start = round( self.opt.width * int(inputs[("dxy_HiS")][i,0]) * 1.0 / int(inputs[("resize_HiS")][i,0]) )
                    y_start = round( self.opt.height * int(inputs[("dxy_HiS")][i,1]) * 1.0 / int(inputs[("resize_HiS")][i,1]) )
                    width_union = round( self.opt.width * self.opt.width * 1.0 / int(inputs[("resize_HiS")][i,0]) )
                    height_union = round( self.opt.height * self.opt.height * 1.0 / int(inputs[("resize_HiS")][i,1]) )
                    disp_union_HiS = F.interpolate(disp_HiS[i].unsqueeze(0), [height_union, width_union], mode="bilinear", align_corners=False)
                    disp_union_MiS = disp_MiS[i,:,y_start:y_start+height_union,x_start:x_start+width_union].unsqueeze(0)
                    mask_disp = mask4consis[i,:,y_start:y_start+height_union,x_start:x_start+width_union].unsqueeze(0)   #[12,1,h,w]
                    disp_diff = self.compute_reprojection_loss(disp_union_MiS, disp_union_HiS)             #[12,1,h,w]
                    disp_diff_mask = disp_diff * mask_disp
                    disp_diff_batch += (disp_diff_mask.sum() / (mask_disp.sum() + 1e-7))
                loss_consis_HiS += disp_diff_batch / self.opt.batch_size

                disp_diff_batch_LoS = 0
                for i in range(self.opt.batch_size):
                    width_un_LoS = int(inputs[("resize_LoS")][i,0])
                    height_un_LoS = int(inputs[("resize_LoS")][i,1])
                    disp_un_LoS = F.interpolate(disp_LoS[i,:,0:height_un_LoS,0:width_un_LoS].unsqueeze(0), [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    disp_un_MiS = disp_MiS[i].unsqueeze(0)
                    disp_diff_LoS = self.compute_reprojection_loss(disp_un_MiS, disp_un_LoS)
                    mask_disp_LoS = mask4consis.unsqueeze(0)
                    disp_diff_LoS_mask = disp_diff_LoS * mask_disp_LoS
                    disp_diff_batch_LoS += (disp_diff_LoS_mask.sum() / (mask_disp_LoS.sum() + 1e-7))
                loss_consis_LoS += disp_diff_batch_LoS / self.opt.batch_size


            losses["loss/{}".format(scale)] = loss_HiS

        loss_HiS /= self.num_scales
        loss_MiS /= self.num_scales
        loss_LoS /= self.num_scales
        loss_consis_HiS /= self.num_scales
        loss_consis_LoS /= self.num_scales
        total_loss = (loss_HiS + loss_MiS + loss_LoS) + 1.0*(loss_consis_HiS + loss_consis_LoS)

        if not self.epoch >= self.opt.epoch_change:
            loss_consis_HiS = torch.tensor(0.0, device=device)
            loss_consis_LoS = torch.tensor(0.0, device=device)

        losses["loss"] = total_loss
        losses["loss_HiS"] = loss_HiS
        losses["loss_MiS"] = loss_MiS
        losses["loss_LoS"] = loss_LoS
        losses["loss_consis_HiS"] = loss_consis_HiS
        losses["loss_consis_LoS"] = loss_consis_LoS
        return losses

    def compute_consistency_loss(self, inputs, disp, disp_2):
        loss_c = 0
        cnt = 1.0
        for i in range(self.opt.batch_size):
            x1, y1 = int(inputs[("dxy")][i,0]), int(inputs[("dxy")][i,1])
            x2, y2 = int(inputs[("dxy_2")][i,0]), int(inputs[("dxy_2")][i,1])
            x_min, x_max = min(x1,x2), max(x1,x2)
            y_min, y_max = min(y1,y2), max(y1,y2)
            if x1==x2 and y1==y2:
                continue
            if abs(x1-x2) >= self.opt.width or abs(y1-y2) >= self.opt.height:
                continue
            x_s, y_s = x_max, y_max
            x_e, y_e = x_min+self.opt.width, y_min+self.opt.height
            if (x_e-x_s)*(y_e-y_s) <= (self.opt.width * self.opt.height)*1.0/4 or (x_e-x_s)*(y_e-y_s) >= (self.opt.width * self.opt.height)*7.0/8:
                continue
            disp_unite_1 = disp[i, 0, abs(y1-y_s):abs(y1-y_e), abs(x1-x_s):abs(x1-x_e)]
            disp_unite_2 = disp_2[i, 0, abs(y2-y_s):abs(y2-y_e), abs(x2-x_s):abs(x2-x_e)]
            diff_disp = ((disp_unite_1 - disp_unite_2).abs() /
                  (disp_unite_1 + disp_unite_2).abs()).clamp(0, 1)
            loss_c += diff_disp.mean()
            cnt += 1

        loss_c /= cnt
        return loss_c


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, lH, lM, lL, cH, cL, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epo {:>3} | bat {:>6} | ex/s: {:5.1f}" + \
            " | lH: {:.4f} | lM: {:.4f} | lL: {:.4f} | cH: {:.10f} | cL: {:.10f} | loss: {:.5f} | te: {} | tl: {} | lr: {}"
        # print(print_string.format(self.epoch, batch_idx, samples_per_sec, lH, lM, lL, cH, cL, loss,
        #                           sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.state_dict()['param_groups'][0]['lr']))
        logging.info(print_string.format(self.epoch, batch_idx, samples_per_sec, lH, lM, lL, cH, cL, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left), self.model_optimizer.state_dict()['param_groups'][0]['lr']))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_HiS_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color_HiS", frame_id, s)][j].data, self.step)
                    writer.add_image(
                        "color_MiS_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color_MiS", frame_id, s)][j].data, self.step)
                    writer.add_image(
                        "color_LoS_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color_LoS", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_HiS_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color_HiS", frame_id, s)][j].data, self.step)
                        
                        writer.add_image(
                            "color_pred_MiS_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color_MiS", frame_id, s)][j].data, self.step)
                        
                        writer.add_image(
                            "color_pred_LoS_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color_LoS", frame_id, s)][j].data, self.step)
                if self.epoch >= self.opt.epoch_change:
                        writer.add_image(
                            "disp_HiS{}/{}".format(s, j),
                            normalize_image(outputs["out_HiS"][("disp", s)][j]), self.step)
                        
                        writer.add_image(
                            "disp_MiS{}/{}".format(s, j),
                            normalize_image(outputs["out_MiS"][("disp", s)][j]), self.step)
                        
                        writer.add_image(
                            "disp_LoS{}/{}".format(s, j),
                            normalize_image(outputs["out_LoS"][("disp", s)][j]), self.step)
                else:
                    writer.add_image(
                            "disp_HiS{}/{}".format(s, j),
                            normalize_image(outputs["out_HiS"][j]), self.step)
                        
                    writer.add_image(
                            "disp_MiS{}/{}".format(s, j),
                            normalize_image(outputs["out_MiS"][("disp", s)][j]), self.step)
                        
                    writer.add_image(
                            "disp_LoS{}/{}".format(s, j),
                            normalize_image(outputs["out_LoS"][j]), self.step)
                    
                    writer.add_image(
                            "out_LoS_mask{}/{}".format(s, j),
                            outputs["out_LoS_mask"][j], self.step)
                    
                    

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_HiS{}/{}".format(s, j),
                        outputs["identity_selection_His/{}".format(s)][j][None, ...], self.step)
                    writer.add_image(
                        "automask_MiS{}/{}".format(s, j),
                        outputs["identity_selection_Mis/{}".format(s)][j][None, ...], self.step)
                    writer.add_image(
                        "automask_LoS{}/{}".format(s, j),
                        outputs["identity_selection_Los/{}".format(s)][j][None, ...], self.step)
                    

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        if self.step % self.save_step != 0:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}_{}".format(self.epoch, self.step / self.save_step))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            # 检查是否是多GPU模型
            if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
                to_save = model.module.state_dict()
            else:
                to_save = model.state_dict()

            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        # 保存优化器状态
        save_path = os.path.join(save_folder, "adam.pth")
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)
        assert os.path.isdir(self.opt.load_weights_folder), "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))
        logging.info("loading model from folder {}".format(self.opt.load_weights_folder))
        
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            
            # 从文件中加载预训练的状态字典
            pretrained_dict = torch.load(path)
            
            # 获取当前模型的状态字典
            model = self.models[n]
            if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                model = model.module
            model_dict = model.state_dict()

            # 筛选出与当前模型匹配的键
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # 更新当前模型的状态字典
            model_dict.update(pretrained_dict)

            # 加载更新后的状态字典
            model.load_state_dict(model_dict, strict=False)


        # 加载优化器状态
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
            new_learning_rate = self.opt.lr_nxt
            for param_group in self.model_optimizer.param_groups:
                param_group["lr"] = new_learning_rate
                
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")