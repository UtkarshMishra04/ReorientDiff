#!/usr/bin/env python

import argparse
import itertools
import json
import time
import pickle
import os
import random
import gdown
from loguru import logger
import numpy as np
import path
import pybullet_planning as pp
import torch
import torch.nn as nn
from tqdm import tqdm
import reorientbot

from reorientbot.examples.reorientation._env_eval_model import Env
from reorientbot.examples.reorientation import _reorient
from reorientbot.examples.reorientation import _utils

from reorientbot.examples.reorientation.reorientable_train import Model as ModelReorient
from reorientbot.examples.reorientation.pickable_train import Model as ModelPickable

save_dir = path.Path(__file__).parent / 'saved_models'
here = path.Path(__file__).abspath().parent
home_data = path.Path("/hddscratch/umishra31/").expanduser()
DEFAULT_DATA_PATH = home_data / 'graspingbot' / 'reorient_dynamic2'

reorient_models = {}

reorient_models["franka_panda/panda_suction"] = ModelReorient()
model_file = gdown.cached_download(
    id="1UsajylR2I0OT31jLRqMA8iNZox1zTKZw",
    path=here
    / "logs/reorientable/20210819_035217.161036-panda_suction/models/model_best-epoch_0142.pt",  # NOQA
    md5="bddb1ee74ea6015cf57fa66e11dabf95",
)
reorient_models["franka_panda/panda_suction"].load_state_dict(torch.load(model_file))
reorient_models["franka_panda/panda_suction"].eval()

pickable_models = {}

pickable_models["franka_panda/panda_suction"] = ModelPickable()
model_file = gdown.cached_download(
    id="11Z1wTicF7i8pcE6nARiJQfkq6VW7dpJH",
    path=here
    / "logs/pickable/20210819_035214.799043-panda_suction/models/model_best-epoch_0050.pt",  # NOQA
    md5="748f277c7b50b507d943a8aa9c9e1476",
)
pickable_models["franka_panda/panda_suction"].load_state_dict(torch.load(model_file))
pickable_models["franka_panda/panda_suction"].eval()

class GradDiscriminator():

    def __init__(self, env, target_grasp_poses, possible_grasp_poses):
        self.env = env
        self.target_grasp_poses = target_grasp_poses
        self.possible_grasp_poses = possible_grasp_poses

        self.model_ro = reorient_models[env._robot_model]
        self.model_ro.cuda()
        self.model_po = pickable_models[env._robot_model]
        self.model_po.cuda()

        obj_to_world = pp.get_pose(env.fg_object_id)
        world_to_obj = pp.invert(obj_to_world)

        # pose representation -> point-normal representation
        target_grasp_points = []
        for grasp_pose in target_grasp_poses:
            ee_to_obj = np.hsplit(grasp_pose, [3])
            grasp_point_start = ee_to_obj[0]
            grasp_point_end = reorientbot.geometry.transform_points(
                [[0, 0, 1]], reorientbot.geometry.transformation_matrix(*ee_to_obj)
            )[0]
            target_grasp_points.append(np.hstack([grasp_point_start, grasp_point_end]))

            if 0:
                pp.draw_pose(
                    np.hsplit(grasp_pose, [3]),
                    parent=env.fg_object_id,
                    length=0.05,
                    width=3,
                )
        target_grasp_points = np.array(target_grasp_points)

        class_ids = [2, 3, 5, 11, 12, 15, 16]

        heightmap = env.obs["pointmap"][:, :, 2]

        object_fg_flags = []
        object_labels = []
        object_poses = []
        for object_id in env.object_ids:
            class_id = _utils.get_class_id(object_id)
            if class_id not in class_ids:
                continue
            object_fg_flags.append(object_id == env.fg_object_id)
            object_label = np.zeros(7)
            object_label[class_ids.index(class_id)] = 1
            object_labels.append(object_label)
            object_poses.append(np.hstack(pp.get_pose(object_id)))
        object_fg_flags = np.stack(object_fg_flags, axis=0).astype(np.float32)
        object_labels = np.stack(object_labels, axis=0).astype(np.float32)
        object_poses = np.stack(object_poses, axis=0).astype(np.float32)

        object_label = object_labels[object_fg_flags == 1][0]
        object_pose = object_poses[object_fg_flags == 1][0]

        if getattr(env, "reverse", False):
            object_pose[:3] = env.PILE_POSITION + [0, -0.4, 0.1]

        self.heightmap = torch.as_tensor(heightmap[None, None]).float().cuda()
        self.object_label = torch.as_tensor(object_label[None]).float().cuda()
        self.object_pose = torch.as_tensor(object_pose[None]).float().cuda()

        self.target_grasp_poses = target_grasp_poses
        self.target_grasp_points = target_grasp_points

    def return_mapped_poses(self, pred_poses):

        # return the closest true pose to the predicted pose for each predicted pose
        # pred_poses: N x 7
        # true_poses: M x 7

        N = pred_poses.shape[0]
        M = self.possible_grasp_poses.shape[0]

        mapped_poses = np.zeros((N, 7))

        for i in range(N):
            dist = np.linalg.norm(pred_poses[i] - self.possible_grasp_poses, axis=1)
            mapped_poses[i] = self.possible_grasp_poses[np.argmin(dist)]

        return mapped_poses

    def calc_grad(self, pred_reorient):

        device = pred_reorient.device

        pick_grasp_poses, reorient_poses = pred_reorient[:, :7], pred_reorient[:, 7:14]

        pick_grasp_poses = pick_grasp_poses.detach().cpu().numpy()
        reorient_poses = reorient_poses.detach().cpu().numpy()

        pick_grasp_poses = self.return_mapped_poses(pick_grasp_poses)

        # pose representation -> point-normal representation
        grasp_points = []
        for grasp_pose in pick_grasp_poses:
            ee_to_obj = np.hsplit(grasp_pose, [3])
            grasp_point_start = ee_to_obj[0]
            grasp_point_end = reorientbot.geometry.transform_points(
                [[0, 0, 1]], reorientbot.geometry.transformation_matrix(*ee_to_obj)
            )[0]
            grasp_points.append(np.hstack([grasp_point_start, grasp_point_end]))

            if 0:
                pp.draw_pose(
                    np.hsplit(grasp_pose, [3]),
                    parent=env.fg_object_id,
                    length=0.05,
                    width=3,
                )
        grasp_points = np.array(grasp_points)

        N_target = self.target_grasp_poses.shape[0]
        N_grasp = grasp_points.shape[0]
        N_reorient = reorient_poses.shape[0]
        B1 = N_grasp * N_reorient
        B2 = N_target * N_reorient
        # logger.info(f"N_grasp: {N_grasp}, N_reorient: {N_reorient}, N_target: {N_target}, B1: {B1}, B2: {B2}")

        reorient_poses = torch.from_numpy(reorient_poses)
        reorient_poses.requires_grad = True

        target_grasp_poses = torch.from_numpy(self.target_grasp_poses)
        target_grasp_points = torch.from_numpy(self.target_grasp_points)
        grasp_poses = torch.from_numpy(pick_grasp_poses)
        grasp_points = torch.from_numpy(grasp_points)

        # pickable pattern is R x G while reorient pattern is G x R
        pickable_grasp_poses = target_grasp_poses.unsqueeze(0).repeat(N_reorient, 1, 1)
        pickable_grasp_poses = pickable_grasp_poses.view(B2, -1).float()
        pickable_grasp_points = target_grasp_points.unsqueeze(0).repeat(N_reorient, 1, 1)
        pickable_grasp_points = pickable_grasp_points.view(B2, -1).float()

        reorientable_grasp_poses = grasp_poses.unsqueeze(1).repeat(1, N_reorient, 1)
        reorientable_grasp_poses = reorientable_grasp_poses.view(B1, -1).float()
        reorientable_grasp_points = grasp_points.unsqueeze(1).repeat(1, N_reorient, 1)
        reorientable_grasp_points = reorientable_grasp_points.view(B1, -1).float()

        prev_loss = 0
        num_hits = 0

        pickable_reorient_poses = reorient_poses.unsqueeze(1).repeat(1, N_target, 1)
        pickable_reorient_poses = pickable_reorient_poses.view(B2, -1).float()

        reorientable_reorient_poses = reorient_poses.unsqueeze(0).repeat(N_grasp, 1, 1)
        reorientable_reorient_poses = reorientable_reorient_poses.view(B1, -1).float()

        pickable_pred = self.model_po(
            heightmap=self.heightmap,
            object_label=self.object_label,
            object_pose=self.object_pose,
            grasp_pose=pickable_grasp_points.unsqueeze(0).cuda(),
            reorient_pose=pickable_reorient_poses.unsqueeze(0).cuda(),
        )

        reorientable_pred, trajectory_length_pred = self.model_ro(
            heightmap=self.heightmap,
            object_label=self.object_label,
            object_pose=self.object_pose,
            grasp_pose=reorientable_grasp_points.unsqueeze(0).cuda(),
            reorient_pose=reorientable_reorient_poses.unsqueeze(0).cuda(),
        )

        pickable_pred = pickable_pred.view(N_reorient, N_target, 1)
        pickable_pred = torch.mean(pickable_pred)
        reorientable_pred_copy = reorientable_pred
        reorientable_pred_copy = reorientable_pred_copy.view(N_grasp, N_reorient, 3)
        reorientable_pred_copy = torch.prod(reorientable_pred_copy, dim=-1)
        reorientable_pred_copy = torch.mean(reorientable_pred_copy)

        # print("pickable_pred", pickable_pred.shape)
        # print("reorientable_pred_copy", reorientable_pred_copy.shape)
        # assert False

        loss = torch.abs(torch.ones_like(pickable_pred) - pickable_pred)**2 + torch.abs(torch.ones_like(reorientable_pred_copy) - reorientable_pred_copy)**2

        grad_value = torch.autograd.grad(loss, reorient_poses)[0] # + 0.1 * torch.randn_like(reorient_poses)

        grad_value = torch.concat([torch.zeros_like(torch.Tensor(pick_grasp_poses)), grad_value, torch.zeros(grad_value.size(0)).unsqueeze(1)], dim=1)

        return grad_value.to(device)

    def last_step_discrimination_reorient(self, pick_grasp_poses, reorient_poses):

        pick_grasp_poses = self.return_mapped_poses(pick_grasp_poses)
        
        # pose representation -> point-normal representation
        grasp_points = []
        for grasp_pose in pick_grasp_poses:
            ee_to_obj = np.hsplit(grasp_pose, [3])
            grasp_point_start = ee_to_obj[0]
            grasp_point_end = reorientbot.geometry.transform_points(
                [[0, 0, 1]], reorientbot.geometry.transformation_matrix(*ee_to_obj)
            )[0]
            grasp_points.append(np.hstack([grasp_point_start, grasp_point_end]))

            if 0:
                pp.draw_pose(
                    np.hsplit(grasp_pose, [3]),
                    parent=env.fg_object_id,
                    length=0.05,
                    width=3,
                )
        grasp_points = np.array(grasp_points)

        N_target = self.target_grasp_poses.shape[0]
        N_grasp = grasp_points.shape[0]
        N_reorient = reorient_poses.shape[0]
        B1 = N_grasp * N_reorient
        B2 = N_target * N_reorient
        logger.info(f"N_grasp: {N_grasp}, N_reorient: {N_reorient}, N_target: {N_target}, B1: {B1}, B2: {B2}")

        reorient_poses = torch.from_numpy(reorient_poses)
        reorient_poses.requires_grad = True

        target_grasp_poses = torch.from_numpy(self.target_grasp_poses)
        target_grasp_points = torch.from_numpy(self.target_grasp_points)
        grasp_poses = torch.from_numpy(pick_grasp_poses)
        grasp_points = torch.from_numpy(grasp_points)

        # pickable pattern is R x G while reorient pattern is G x R
        pickable_grasp_poses = target_grasp_poses.unsqueeze(0).repeat(N_reorient, 1, 1)
        pickable_grasp_poses = pickable_grasp_poses.view(B2, -1).float()
        pickable_grasp_points = target_grasp_points.unsqueeze(0).repeat(N_reorient, 1, 1)
        pickable_grasp_points = pickable_grasp_points.view(B2, -1).float()

        reorientable_grasp_poses = grasp_poses.unsqueeze(1).repeat(1, N_reorient, 1)
        reorientable_grasp_poses = reorientable_grasp_poses.view(B1, -1).float()
        reorientable_grasp_points = grasp_points.unsqueeze(1).repeat(1, N_reorient, 1)
        reorientable_grasp_points = reorientable_grasp_points.view(B1, -1).float()

        prev_loss = 0
        num_hits = 0

        while True:

            pickable_reorient_poses = reorient_poses.unsqueeze(1).repeat(1, N_target, 1)
            pickable_reorient_poses = pickable_reorient_poses.view(B2, -1).float()

            reorientable_reorient_poses = reorient_poses.unsqueeze(0).repeat(N_grasp, 1, 1)
            reorientable_reorient_poses = reorientable_reorient_poses.view(B1, -1).float()

            pickable_pred = self.model_po(
                heightmap=self.heightmap,
                object_label=self.object_label,
                object_pose=self.object_pose,
                grasp_pose=pickable_grasp_points.unsqueeze(0).cuda(),
                reorient_pose=pickable_reorient_poses.unsqueeze(0).cuda(),
            )

            reorientable_pred, trajectory_length_pred = self.model_ro(
                heightmap=self.heightmap,
                object_label=self.object_label,
                object_pose=self.object_pose,
                grasp_pose=reorientable_grasp_points.unsqueeze(0).cuda(),
                reorient_pose=reorientable_reorient_poses.unsqueeze(0).cuda(),
            )

            pickable_pred = pickable_pred.view(N_reorient, N_target, 1)
            pickable_pred = torch.mean(pickable_pred)
            reorientable_pred_copy = reorientable_pred
            reorientable_pred_copy = reorientable_pred_copy.view(N_grasp, N_reorient, 3)
            reorientable_pred_copy = torch.prod(reorientable_pred_copy, dim=-1)
            reorientable_pred_copy = torch.mean(reorientable_pred_copy)

            loss = torch.abs(torch.ones_like(pickable_pred) - pickable_pred)**2 + torch.abs(torch.ones_like(reorientable_pred_copy) - reorientable_pred_copy)**2

            grad_value = torch.autograd.grad(loss, reorient_poses)[0]

            reorient_poses = reorient_poses - 0.01 * grad_value - 0.001 * torch.randn_like(reorient_poses)

            # if abs(loss.item() - prev_loss) < 1e-3:
            #     num_hits += 1
            # else:
            #     num_hits = 0

            # prev_loss = loss.item()

            num_hits += 1

            if num_hits > 10:
                break

        pickable_reorient_poses = reorient_poses.unsqueeze(1).repeat(1, N_target, 1)
        pickable_reorient_poses = pickable_reorient_poses.view(B2, -1).float()

        reorientable_reorient_poses = reorient_poses.unsqueeze(0).repeat(N_grasp, 1, 1)
        reorientable_reorient_poses = reorientable_reorient_poses.view(B1, -1).float()

        reorient_poses = reorientable_reorient_poses.detach().cpu().numpy()
        pick_grasp_poses = reorientable_grasp_poses.detach().cpu().numpy()
        trajectory_length_pred = trajectory_length_pred.detach().cpu().numpy()[0]
        reorientable_pred = reorientable_pred.detach().cpu().numpy()[0]

        pick_grasp_poses = pick_grasp_poses.reshape(N_grasp, N_reorient, 7)
        reorient_poses = reorient_poses.reshape(N_grasp, N_reorient, 7)
        trajectory_length_pred = trajectory_length_pred.reshape(
            N_grasp, N_reorient
        )
        reorientable_pred = reorientable_pred.reshape(N_grasp, N_reorient, 3)
        reorientable_pred = np.prod(reorientable_pred, axis=-1)

        N_top = 3
        i_grasp = np.arange(N_grasp)[:, None].repeat(N_top, axis=1)
        i_reorient = np.argsort(reorientable_pred, axis=1)[:, -N_top:]

        reorientable_pred = reorientable_pred[i_grasp, i_reorient]
        trajectory_length_pred = trajectory_length_pred[i_grasp, i_reorient]
        pick_grasp_poses = pick_grasp_poses[i_grasp, i_reorient]
        reorient_poses = reorient_poses[i_grasp, i_reorient]

        reorientable_pred = reorientable_pred.reshape(N_grasp * N_top)
        trajectory_length_pred = trajectory_length_pred.reshape(N_grasp * N_top)
        pick_grasp_poses = pick_grasp_poses.reshape(N_grasp * N_top, 7)
        reorient_poses = reorient_poses.reshape(N_grasp * N_top, 7)

        indices = np.argsort(reorientable_pred)[::-1]

        return pick_grasp_poses[indices], reorient_poses[indices]

    def order_pickability(self, reorient_poses):


        N_target = self.target_grasp_poses.shape[0]
        N_reorient = reorient_poses.shape[0]
        B2 = N_target * N_reorient
        logger.info(f"N_reorient: {N_reorient}, N_target: {N_target}B2: {B2}")

        reorient_poses = torch.from_numpy(reorient_poses)

        target_grasp_poses = torch.from_numpy(self.target_grasp_poses)
        target_grasp_points = torch.from_numpy(self.target_grasp_points)

        # pickable pattern is R x G while reorient pattern is G x R
        pickable_grasp_poses = target_grasp_poses.unsqueeze(0).repeat(N_reorient, 1, 1)
        pickable_grasp_poses = pickable_grasp_poses.view(B2, -1).float()
        pickable_grasp_points = target_grasp_points.unsqueeze(0).repeat(N_reorient, 1, 1)
        pickable_grasp_points = pickable_grasp_points.view(B2, -1).float()

        pickable_reorient_poses = reorient_poses.unsqueeze(1).repeat(1, N_target, 1)
        pickable_reorient_poses = pickable_reorient_poses.view(B2, -1).float()

        pickable_pred = self.model_po(
            heightmap=self.heightmap,
            object_label=self.object_label,
            object_pose=self.object_pose,
            grasp_pose=pickable_grasp_points.unsqueeze(0).cuda(),
            reorient_pose=pickable_reorient_poses.unsqueeze(0).cuda(),
        )

        pickable_pred = pickable_pred.view(N_reorient, N_target).detach().cpu().numpy()
        pickable_grasp_poses = pickable_grasp_poses.view(N_reorient, N_target, -1).detach().cpu().numpy()
        pickable_reorient_poses = pickable_reorient_poses.view(N_reorient, N_target, -1).detach().cpu().numpy()

        pickable_pred_reo = np.mean(pickable_pred, axis=1)
        indices_reo = np.argsort(pickable_pred_reo, axis=0)[::-1]

        pickable_pred_gra = np.mean(pickable_pred, axis=0)
        indices_gra = np.argsort(pickable_pred_gra, axis=0)[::-1]

        return pickable_reorient_poses[indices_reo, 0, :]

