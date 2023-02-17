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
from reorientbot.examples.reorientation.pickable_eval import (
    get_goal_oriented_reorient_poses,  # NOQA
)
from reorientbot.examples.reorientation.pickable_reorient_poses import (
    get_reorient_poses,  # NOQA
)
from reorientbot.examples.reorientation.reorientable_train import Model as ModelReorient
from reorientbot.examples.reorientation.pickable_train import Model as ModelPickable

from models.language.hf_encoding import HuggingFaceEncoding
from models.language.utils.joint_model import PoseObjPredictor
from models.diffusion.unet import GraspNet
from algorithm.diffusion.cond_diffusion import VarianceSchedule, DiffusionGrasps
from dataset.only_reorientation_data import get_data_loader

save_dir = path.Path(__file__).parent / 'saved_models'
here = path.Path(__file__).abspath().parent
home_data = path.Path("/hddscratch/umishra31/").expanduser()
DEFAULT_DATA_PATH = home_data / 'graspingbot' / 'reorient_dynamic2'

POSE_MODEL_PATH = save_dir / 'hf_diff_poseobjnet_2023-02-02_21-18-58' / 'final_model.pt'
REORIENT_MODEL_PATH = save_dir / 'hf_diff_reorient_diff_2023-02-04_23-19-54'

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

def grad_adjust_poses(env, target_grasp_poses, pick_grasp_poses, reorient_poses):

    print("target_grasp_poses", target_grasp_poses.shape)
    print("pick_grasp_poses", pick_grasp_poses.shape)
    print("reorient_poses", reorient_poses.shape)

    model_ro = reorient_models[env._robot_model]
    model_ro.cuda()
    model_po = pickable_models[env._robot_model]
    model_po.cuda()

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

    N_target = target_grasp_poses.shape[0]
    N_grasp = grasp_points.shape[0]
    N_reorient = reorient_poses.shape[0]
    B1 = N_grasp * N_reorient
    B2 = N_target * N_reorient
    logger.info(f"N_grasp: {N_grasp}, N_reorient: {N_reorient}, N_target: {N_target}, B1: {B1}, B2: {B2}")

    reorient_poses = torch.from_numpy(reorient_poses)
    reorient_poses.requires_grad = True

    target_grasp_poses = torch.from_numpy(target_grasp_poses)
    target_grasp_points = torch.from_numpy(target_grasp_points)
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

        pickable_pred = model_po(
            heightmap=torch.as_tensor(heightmap[None, None]).float().cuda(),
            object_label=torch.as_tensor(object_label[None]).float().cuda(),
            object_pose=torch.as_tensor(object_pose[None]).float().cuda(),
            grasp_pose=pickable_grasp_points.unsqueeze(0).cuda(),
            reorient_pose=pickable_reorient_poses.unsqueeze(0).cuda(),
        )

        reorientable_pred, trajectory_length_pred = model_ro(
            heightmap=torch.as_tensor(heightmap[None, None]).float().cuda(),
            object_label=torch.as_tensor(object_label[None]).float().cuda(),
            object_pose=torch.as_tensor(object_pose[None]).float().cuda(),
            grasp_pose=reorientable_grasp_points.unsqueeze(0).cuda(),
            reorient_pose=reorientable_reorient_poses.unsqueeze(0).cuda(),
        )

        pickable_pred = pickable_pred.view(N_reorient, N_target, 1)
        pickable_pred = torch.mean(pickable_pred)
        reorientable_pred = reorientable_pred.view(N_grasp, N_reorient, 3)
        reorientable_pred = torch.prod(reorientable_pred, dim=-1)
        reorientable_pred = torch.mean(reorientable_pred)

        loss = torch.abs(torch.ones_like(pickable_pred) - pickable_pred) + torch.abs(torch.ones_like(reorientable_pred) - reorientable_pred)**2

        grad_value = torch.autograd.grad(loss, reorient_poses)[0]

        reorient_poses = reorient_poses - 0.01 * grad_value

        if (loss.item() - prev_loss).abs() < 1e-3:
            num_hits += 1
        else:
            num_hits = 0

        prev_loss = loss.item()

        if num_hits > 100:
            break

    target_grasp_poses = target_grasp_poses[None, :, :].repeat(N_reorient, axis=1)
    target_grasp_points = target_grasp_points[None, :, :].repeat(N_reorient, axis=1)
    target_reorient_poses = reorient_poses[:, None, :].repeat(N_target, axis=0)
    pick_grasp_poses = pick_grasp_poses[:, None, :].repeat(N_reorient, axis=1)
    grasp_points = grasp_points[:, None, :].repeat(N_reorient, axis=1)
    reorient_poses = reorient_poses[None, :, :].repeat(N_grasp, axis=0)

    pick_grasp_poses = pick_grasp_poses.reshape(B1, -1).astype(np.float32)
    grasp_points = grasp_points.reshape(B1, -1).astype(np.float32)
    reorient_poses = reorient_poses.reshape(B1, -1).astype(np.float32)
    target_grasp_poses = target_grasp_poses.reshape(B2, -1).astype(np.float32)
    target_grasp_points = target_grasp_points.reshape(B2, -1).astype(np.float32)
    target_reorient_poses = target_reorient_poses.reshape(B2, -1).astype(np.float32)

    grasp_points = torch.as_tensor(grasp_points[None])
    reorient_poses = torch.as_tensor(reorient_poses[None])
    target_grasp_points = torch.as_tensor(target_grasp_points[None])
    target_reorient_poses = torch.as_tensor(target_reorient_poses[None])

    # reorient_poses.requires_grad = True
    target_reorient_poses.requires_grad = True
    
    pickable_pred_copy = 0
    old_pickable_pred_copy = np.inf
    change_pick = torch.zeros_like(target_reorient_poses)
    num_trials = 0

    while num_trials < 100: # abs(pickable_pred_copy - old_pickable_pred_copy) > 0.00001: # or pickable_pred_copy < 0.99:

        old_pickable_pred_copy = pickable_pred_copy

        pickable_pred = model_po(
            heightmap=torch.as_tensor(heightmap[None, None]).float().cuda(),
            object_label=torch.as_tensor(object_label[None]).float().cuda(),
            object_pose=torch.as_tensor(object_pose[None]).float().cuda(),
            grasp_pose=torch.as_tensor(target_grasp_points).float().cuda(),
            reorient_pose=torch.as_tensor(target_reorient_poses).float().cuda(),
        )

        pickable_pred_copy = torch.mean(pickable_pred)

        grad_value = torch.autograd.grad(
            torch.ones_like(pickable_pred_copy) - pickable_pred_copy,
            target_reorient_poses,
            grad_outputs=torch.ones_like(pickable_pred_copy),
        )[0]

        # update reorient_poses with grad_value
        target_reorient_poses = target_reorient_poses - (1 * grad_value + 0.9*change_pick)
        change_pick = 1 * grad_value + 0.9*change_pick

        pickable_pred_copy = pickable_pred_copy.detach().cpu().numpy()

        print("pickable_pred_copy", pickable_pred_copy)

        if abs(pickable_pred_copy - old_pickable_pred_copy) < 0.0001:
            num_trials += 1
        else:
            num_trials = 0
        
    target_reorient_poses = target_reorient_poses.detach().cpu().numpy()[0]
    target_reorient_poses = target_reorient_poses.reshape(N_reorient, N_target, -1)[:, 0, :]

    reorientable_pred_copy = 0
    old_reorientable_pred_copy = np.inf
    change_reorient = torch.zeros_like(reorient_poses)

    # for i in tqdm(range(1000)):
    while abs(reorientable_pred_copy - old_reorientable_pred_copy) > 0.0001: # or reorientable_pred_copy < 0.99:

        old_reorientable_pred_copy = reorientable_pred_copy

        # with torch.no_grad():
        reorientable_pred, trajectory_length_pred = model_ro(
            heightmap=torch.as_tensor(heightmap[None, None]).float().cuda(),
            object_label=torch.as_tensor(object_label[None]).float().cuda(),
            object_pose=torch.as_tensor(object_pose[None]).float().cuda(),
            grasp_pose=grasp_points.float().cuda(),
            reorient_pose=reorient_poses.float().cuda(),
        )

        reorientable_pred_copy = torch.prod(reorientable_pred, dim=-1)
        reorientable_pred_copy = torch.mean(reorientable_pred_copy)

        grad_value = torch.autograd.grad(
            torch.ones_like(reorientable_pred_copy) - reorientable_pred_copy,
            reorient_poses,
            grad_outputs=torch.ones_like(reorientable_pred_copy),
        )[0]

        # update reorient_poses with grad_value
        reorient_poses = reorient_poses - (1 * grad_value + 0.9*change_reorient)
        change_reorient = 1 * grad_value + 0.9*change_reorient

        reorientable_pred_copy = reorientable_pred_copy.detach().cpu().numpy()
        reorientable_pred = reorientable_pred.detach().cpu().numpy()
        reorientable_pred = reorientable_pred.reshape(N_grasp, N_reorient, 3)
        reorientable_pred = np.prod(reorientable_pred, axis=2)

        print(reorientable_pred.mean(), reorientable_pred_copy)

    # print(grad_value.shape)
    # assert False

    reorient_poses = reorient_poses.detach().cpu().numpy()[0]
    trajectory_length_pred = trajectory_length_pred.detach().cpu().numpy()[0]

    pick_grasp_poses = pick_grasp_poses.reshape(N_grasp, N_reorient, 7)
    reorient_poses = reorient_poses.reshape(N_grasp, N_reorient, 7)
    trajectory_length_pred = trajectory_length_pred.reshape(
        N_grasp, N_reorient
    )

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

    print(indices, pick_grasp_poses.shape, reorient_poses.shape)

    return pick_grasp_poses[indices], reorient_poses[indices]


def plan_dynamic_reorient(env, grasp_poses, reorient_poses):

    obj_to_world = pp.get_pose(env.fg_object_id)
    world_to_obj = pp.invert(obj_to_world)
    
    N_grasps = grasp_poses.shape[0]
    N_reorients = reorient_poses.shape[0]

    result = {}
    # for indexg in tqdm(range(N_grasps)):
    #     ee_to_obj = np.hsplit(grasp_poses[indexg], [3])
    #     ee_to_world = pp.multiply(obj_to_world, ee_to_obj)
    #     obj_af_to_world = np.hsplit(reorient_poses[indexg], [3])

    #     result = _reorient.plan_reorient(
    #         env, np.hstack(ee_to_world), np.hstack(obj_af_to_world)
    #     )

    #     if _utils.get_class_id(env.fg_object_id) == 11:
    #         if "js_place_length" in result and result["js_place_length"] > 4.5:
    #             result.pop("js_place")
    #             result.pop("js_place_length")

    #     if "js_place" in result:
    #         logger.success("Success")
    #         break
    #     else:
    #         logger.warning("Failed")

    for indexg in tqdm(range(N_grasps)):
        for indexr in range(N_reorients):
            ee_to_obj = np.hsplit(grasp_poses[indexg], [3])
            ee_to_world = pp.multiply(obj_to_world, ee_to_obj)
            obj_af_to_world = np.hsplit(reorient_poses[indexr], [3])

            result = _reorient.plan_reorient(
                env, np.hstack(ee_to_world), np.hstack(obj_af_to_world)
            )

            if _utils.get_class_id(env.fg_object_id) == 11:
                if "js_place_length" in result and result["js_place_length"] > 4.5:
                    result.pop("js_place")
                    result.pop("js_place_length")

            if "js_place" in result:
                logger.success("Success")
                break
            else:
                logger.warning("Failed")

        if "js_place" in result:
            logger.success("Success")
            break
        else:
            logger.warning("Failed")

    reorient_data_dict = {}

    return result, reorient_data_dict

def return_mapped_poses(pred_poses, true_poses):

    # return the closest true pose to the predicted pose for each predicted pose
    # pred_poses: N x 7
    # true_poses: M x 7

    N = pred_poses.shape[0]
    M = true_poses.shape[0]

    mapped_poses = np.zeros((N, 7))

    for i in range(N):
        dist = np.linalg.norm(pred_poses[i] - true_poses, axis=1)
        mapped_poses[i] = true_poses[np.argmin(dist)]

    return mapped_poses

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--robot-model",
        default="franka_panda/panda_suction",
        choices=["franka_panda/panda_suction", "franka_panda/panda_drl"],
        help="robot model",
    )
    parser.add_argument("--seed", type=int, help="seed", required=True)
    parser.add_argument("--sample", type=int, help="sample_num", default=None)
    parser.add_argument(
        "--face",
        choices=["front", "back", "right", "left"],
        default="front",
        help="face",
    )
    parser.add_argument("--level", type=int, choices=[1, 2, 3], help="level", default=2)
    parser.add_argument("--mp4", help="mp4")
    parser.add_argument("--nogui", action="store_true", help="no gui")
    parser.add_argument("--nodebug", action="store_true", help="no debug")
    args = parser.parse_args()

    all_class_ids = [2, 3, 5, 11, 12, 15]

    # Load the model and the dataset

    import json
    with open(REORIENT_MODEL_PATH / 'config.txt', 'r') as f:
        config = json.load(f)

    # load dataset
    batch_size = 1
    train_loader, test_loader, dataset = get_data_loader(batch_size=batch_size, data_path=DEFAULT_DATA_PATH)

    image_shape = config['image_shape']
    output_dim = config['output_dim']
    rgbd_shape = config['rgbd_shape']
    pose_dim = config['pose_dim']
    object_dim = config['object_dim']
    num_grasps = config['num_grasps']
    grasp_dim = config['grasp_dim']
    num_reorientations = config['num_reorientations']
    reorient_dim = config['reorient_dim']
    learning_rate = config['learning_rate']
    condition_dim = config['condition_dim']
    num_epochs = config['epochs']
    num_train_grasps = config['num_train_grasps']
    pose_hdims = list(config['pose_hdims'])
    object_hdims = list(config['object_hdims'])

    # load model
    model = HuggingFaceEncoding()
    model.eval()

    pose_model = PoseObjPredictor(image_shape, pose_dim, object_dim, output_dim, pose_hdims, object_hdims)
    pose_model = nn.DataParallel(pose_model)
    pose_model.load_state_dict(torch.load(POSE_MODEL_PATH))
    pose_model = pose_model.to('cuda')
    pose_model.eval()

    diffusion_unet = GraspNet(
        num_grasps=num_reorientations,
        grasp_dim=reorient_dim,
        condition_dim = output_dim + output_dim + output_dim,
    ).to('cuda')

    diffusion_model = DiffusionGrasps(
        net=diffusion_unet
    ).to('cuda')
    diffusion_model.load_state_dict(torch.load(REORIENT_MODEL_PATH  / 'final_diff_model.pt'))
    diffusion_model.eval()

    num_samples = len(dataset)

    num_trials = 10
    num_successful_trials = 0


    for trail in range(num_trials):
        if args.sample is not None:
            sample_num = args.sample
        else:
            sample_num = np.random.randint(0, num_samples)

        sample_num = np.random.randint(0, num_samples)

        sample_data = dataset[sample_num]

        input_image = sample_data[0]
        input_text = sample_data[1]
        input_rgbd = sample_data[3]
        actual_pose = sample_data[2].numpy()
        actual_object = sample_data[4].numpy()
        actual_successful_grasp_poses = sample_data[5].numpy()
        actual_reorientation_data = sample_data[6].numpy()

        actual_pose, actual_successful_grasp_poses, actual_reorientation_data, actual_object = dataset.parse_prediction(
            actual_pose, actual_successful_grasp_poses, actual_reorientation_data, actual_object
        )

        actual_pose = np.hsplit(actual_pose, [3])

        sample_dict = {
            "pile_file": sample_data[8],
            "true_place_pose": actual_pose,
            "true_target_class_id": actual_object,
            "succesful_grasp_poses": actual_successful_grasp_poses,
            "reorient_data": {
                "grasp_poses": actual_reorientation_data[:, :7],
                "reorient_poses": actual_reorientation_data[:, 7:14],
            }
        }

        with torch.no_grad():
            image_batch = torch.Tensor(input_image).unsqueeze(0)
            text_batch = [input_text]
            clip_image, clip_text = model.get_image_features(image_batch), model.get_text_features(text_batch)

            clip_image = clip_image.to('cuda')
            clip_text = clip_text.to('cuda')
            rgbd_batch = torch.Tensor(input_rgbd).unsqueeze(0).to('cuda')

            diffusion_condition = pose_model.module.get_embedding(clip_image, clip_text, rgbd_batch)

            pred_pose, pred_orientation, pred_object, pred_mask = pose_model(clip_image, clip_text, rgbd_batch)
            sampled_reorients = diffusion_model.sample(condition=diffusion_condition, num_grasps=num_reorientations*2, grasp_dim=reorient_dim)

        predicted_pose = torch.cat((pred_pose, torch.zeros_like(pred_orientation), pred_orientation), dim=1)[0].detach().cpu().numpy()
        predicted_object = pred_object[0].detach().cpu().numpy()

        predicted_pose, _, sampled_reorients, pred_object = dataset.parse_prediction(predicted_pose, None, sampled_reorients[0], predicted_object)

        predicted_pose = np.hsplit(predicted_pose, [3])
        predicted_pose[1] = actual_pose[1] #np.zeros_like(predicted_pose[1])

        # if "front" in input_text:
        #     predicted_pose[1][0] = 1
        # elif "back" in input_text:
        #     predicted_pose[1][1] = 1
        # elif "left" in input_text:
        #     predicted_pose[1][2] = 1
        # elif "right" in input_text:
        #     predicted_pose[1][3] = 1

        sample_dict["place_pose"] = predicted_pose #actual_pose #predicted_pose 63 78 83
        sample_dict["target_class_id"] = pred_object

        sample_dict["pred_reorient_data"] = {
            "grasp_poses": sampled_reorients[:, :7],
            "reorient_poses": sampled_reorients[:, 7:14],
        }

        pred_segmentation_image = pred_mask[0].detach().cpu().numpy().transpose(1, 2, 0)

        # process predicted segmentation image as 1 for value greater than 50% of max and else 0
        pred_segmentation_image = np.where(pred_segmentation_image > 0.5 * np.max(pred_segmentation_image), 1, 0)

        # rgb_image = input_image.detach().cpu().numpy().transpose(1, 2, 0)
        # rgbd_image = input_rgbd.detach().cpu().numpy().transpose(1, 2, 0)
        # depth_image = rgbd_image[:, :, 3]
        # segmentation_image = sample_data[7].detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # # show all images together with text
        # fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        # ax[0, 0].imshow(rgb_image)
        # ax[0, 0].set_title('RGB Image')
        # ax[0, 1].imshow(depth_image)
        # ax[0, 1].set_title('Depth Image')
        # ax[1, 0].imshow(segmentation_image)
        # ax[1, 0].set_title('Segmentation Image')
        # ax[1, 1].imshow(pred_segmentation_image)
        # ax[1, 1].set_title('Predicted Segmentation Image')
        # fig.suptitle(input_text)
        # plt.savefig('images/analysis_{}.png'.format(sample_num))
        # assert False

        try:

            env = Env(
                class_ids=all_class_ids,
                robot_model=args.robot_model,
                gui=not args.nogui,
                mp4=args.mp4,
                face=args.face,
                level=args.level,
                debug=not args.nodebug,
                config=sample_dict
            )
            env.random_state = np.random.RandomState(args.seed)
            env.eval = True
            env.reset()

            print("Sample Number: {}".format(sample_num))
            print("Text: {}".format(input_text))
            print("Pose: {} || {}".format(env.PLACE_POSE, predicted_pose))
            print("Actual Pose: {}".format(sample_dict["true_place_pose"]))
            print("Object: {}".format(pred_object))
            print("Actual Object: {}".format(sample_dict["true_target_class_id"]))

            t_start = time.time()

            # succesful_grasp_poses = [sample[0] for sample in sample_dict["succesful_grasp_poses"]]
            index = np.random.choice(sample_dict["succesful_grasp_poses"].shape[0], 4)
            succesful_grasp_poses = sample_dict["succesful_grasp_poses"][index]

            (
                _,
                pickable,
                succesful_grasp_poses,
            ) = get_goal_oriented_reorient_poses(env)

            # possible_grasp_poses = np.array(
            #     list(itertools.islice(_reorient.get_grasp_poses_eval(env, pred_segmentation_image), 100))
            # )

            possible_grasp_poses = np.array(
                list(itertools.islice(_reorient.get_grasp_poses(env), 100))
            )

            # convert to w.r.t. object from world
            transformed_grasp_poses = []

            obj_to_world = pp.get_pose(env.fg_object_id)
            world_to_obj = pp.invert(obj_to_world)

            for grasp_pose in possible_grasp_poses:
                grasp_pose = np.hsplit(grasp_pose, [3])
                grasp_pose = pp.multiply(world_to_obj, grasp_pose)
                transformed_grasp_poses.append(np.hstack(grasp_pose))

            possible_grasp_poses = np.array(transformed_grasp_poses)

            possible_reorient_poses = get_reorient_poses(env)
            possible_reorient_poses = possible_reorient_poses[np.random.choice(possible_reorient_poses.shape[0], 1000, replace=False)]

            success = False
            is_reorientation_needed = True

            grasp_poses = sample_dict["pred_reorient_data"]["grasp_poses"]
            reorient_poses = sample_dict["pred_reorient_data"]["reorient_poses"]

            grasp_poses, reorient_poses = grad_adjust_poses(env, succesful_grasp_poses, grasp_poses, reorient_poses)

            # transformed_grasp_poses = []

            # for grasp_pose in grasp_poses:
            #     obj_to_world = pp.get_pose(env.fg_object_id)
            #     grasp_pose = np.hsplit(grasp_pose, [3])
            #     grasp_pose = pp.multiply(obj_to_world, grasp_pose)
            #     transformed_grasp_poses.append(np.hstack(grasp_pose))

            grasp_poses = return_mapped_poses(grasp_poses, possible_grasp_poses)

            # true_transformed_grasp_poses = []

            # for grasp_pose in grasp_poses:
            #     obj_to_world = pp.get_pose(env.fg_object_id)
            #     grasp_pose = np.hsplit(grasp_pose, [3])
            #     grasp_pose = pp.multiply(obj_to_world, grasp_pose)
            #     true_transformed_grasp_poses.append(np.hstack(grasp_pose))

            # for grasp_pose in possible_grasp_poses:
            #     line = reorientbot.pybullet.draw_lines_from_poses(np.hstack(grasp_pose), color=(0, 0, 1))

            for grasp_pose in grasp_poses:
                grasp_pose = np.hsplit(grasp_pose, [3])
                grasp_pose = pp.multiply(obj_to_world, grasp_pose)
                line = reorientbot.pybullet.draw_lines_from_poses(np.hstack(grasp_pose))

            for grasp_pose in possible_grasp_poses:
                grasp_pose = np.hsplit(grasp_pose, [3])
                grasp_pose = pp.multiply(obj_to_world, grasp_pose)
                line = reorientbot.pybullet.draw_lines_from_poses(np.hstack(grasp_pose))

            reorient_poses = return_mapped_poses(reorient_poses, possible_reorient_poses)
            
            for reorient_pose in reorient_poses:
                obj_af = reorientbot.pybullet.duplicate(
                    env.fg_object_id,
                    collision=False,
                    rgba_color=(1, 1, 1, 0.5),
                    position=reorient_pose[:3],
                    quaternion=reorient_pose[3:],
                    mass=0,
                )

            # assert False

            # reorientbot.pybullet.step_and_sleep()


            # choose random
            index = np.random.choice(grasp_poses.shape[0], 10)
            grasp_poses = grasp_poses[index] #return_mapped_poses(grasp_poses[index], possible_grasp_poses)
            reorient_poses = reorient_poses[index] #return_mapped_poses(reorient_poses[index], possible_reorient_poses)

            result, _ = plan_dynamic_reorient(
                env, grasp_poses, reorient_poses
            )

            planning_time = time.time() - t_start

            if "js_place" not in result:
                logger.error("No solution is found")
                success_reorient = False
                execution_time = np.nan
                trajectory_length = np.nan
                success = False
            else:
                exec_result = _reorient.execute_reorient(env, result)
                success_reorient = True
                execution_time = exec_result["t_place"]
                trajectory_length = result["js_place_length"]

                for _ in range(480):
                    pp.step_simulation()
                    if pp.has_gui():
                        time.sleep(pp.get_time_step())

                result = _reorient.plan_place_eval(env, succesful_grasp_poses)
                if "js_place" not in result:
                    logger.error("Failed to plan pick-and-place for {}".format(env.fg_class_name))
                    success = False
                else:
                    _reorient.execute_place(env, result)
                    obj_to_world_target = env.PLACE_POSE
                    obj_to_world_actual = pp.get_pose(env.fg_object_id)
                    pcd_file = reorientbot.datasets.ycb.get_pcd_file(
                        _utils.get_class_id(env.fg_object_id)
                    )
                    pcd_obj = np.loadtxt(pcd_file)
                    pcd_target = reorientbot.geometry.transform_points(
                        pcd_obj,
                        reorientbot.geometry.transformation_matrix(
                            *obj_to_world_target
                        ),
                    )
                    pcd_actual = reorientbot.geometry.transform_points(
                        pcd_obj,
                        reorientbot.geometry.transformation_matrix(
                            *obj_to_world_actual
                        ),
                    )
                    auc = reorientbot.geometry.average_distance_auc(
                        pcd_target, pcd_actual, max_threshold=0.1
                    )
                    success = float(auc) > 0.9

            if success:
                num_successful_trials += 1

        except Exception as e:
            print(e)
            assert False

    print("Number of successful trials: {} out of {}".format(num_successful_trials, num_trials))


if __name__ == "__main__":
    main()
