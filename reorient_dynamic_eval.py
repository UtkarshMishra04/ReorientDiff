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
from tqdm import tqdm
import reorientbot

from reorientbot.examples.reorientation._env_eval import Env
from reorientbot.examples.reorientation import _reorient
from reorientbot.examples.reorientation import _utils
from reorientbot.examples.reorientation.pickable_eval import (
    get_goal_oriented_reorient_poses,  # NOQA
)
from reorientbot.examples.reorientation.reorientable_train import Model


here = path.Path(__file__).abspath().parent
home_data = path.Path("/hddscratch/umishra31/").expanduser()
DEFAULT_DATA_PATH = home_data / 'graspingbot' / 'reorient_dynamic2'

models = {}

models["franka_panda/panda_drl"] = Model()
model_file = gdown.cached_download(
    id="1phRRjEqCelMo8W2O5_kYmjrmV8DSMtSe",
    path=here
    / "logs/reorientable/20210811_174552.543384-panda_drl/models/model_best-epoch_0111.pt",  # NOQA
    md5="b784e2b7c866f9475d7a38d74d321e3b",
)
models["franka_panda/panda_drl"].load_state_dict(torch.load(model_file))
models["franka_panda/panda_drl"].eval()

models["franka_panda/panda_suction"] = Model()
model_file = gdown.cached_download(
    id="1UsajylR2I0OT31jLRqMA8iNZox1zTKZw",
    path=here
    / "logs/reorientable/20210819_035217.161036-panda_suction/models/model_best-epoch_0142.pt",  # NOQA
    md5="bddb1ee74ea6015cf57fa66e11dabf95",
)
models["franka_panda/panda_suction"].load_state_dict(torch.load(model_file))
models["franka_panda/panda_suction"].eval()

def plan_dynamic_reorient(env, grasp_poses, reorient_poses, pickable):
    model = models[env._robot_model]
    model.cuda()

    obj_to_world = pp.get_pose(env.fg_object_id)
    world_to_obj = pp.invert(obj_to_world)
    
    N_grasps = grasp_poses.shape[0]

    result = {}
    for index in tqdm(range(N_grasps)):
        ee_to_obj = np.hsplit(grasp_poses[index], [3])
        ee_to_world = pp.multiply(obj_to_world, ee_to_obj)
        obj_af_to_world = np.hsplit(reorient_poses[index], [3])

        result = _reorient.plan_reorient(
            env, np.hstack(ee_to_world), np.hstack(obj_af_to_world)
        )

        if _utils.get_class_id(env.fg_object_id) == 11:
            if "js_place_length" in result and result["js_place_length"] > 4.5:
                result.pop("js_place")
                result.pop("js_place_length")

        if "js_place" in result:
            logger.success(
                f"pickable={pickable[index]:.1%}, "
                # f"graspable_pred={reorientable_pred[index, 0]:.1%}, "
                # f"placable_pred={reorientable_pred[index, 1]:.1%}, "
                # f"reorientable_pred={reorientable_pred[index, 2]:.1%}, "
                # f"reorientable_pred={reorientable_pred[index]:.1%}, "
                # f"trajectory_length_pred={trajectory_length_pred[index]:.1f}, "
                # f"trajectory_length_true={result['js_place_length']:.1f}"
            )
            break
        else:
            logger.warning(
                f"pickable={pickable[index]:.1%}, "
                # f"graspable_pred={reorientable_pred[index, 0]:.1%}, "
                # f"placable_pred={reorientable_pred[index, 1]:.1%}, "
                # f"reorientable_pred={reorientable_pred[index, 2]:.1%}, "
                # f"reorientable_pred={reorientable_pred[index]:.1%}, "
                # f"trajectory_length_pred={trajectory_length_pred[index]:.1f}"
            )

    reorient_data_dict = {}

    return result, reorient_data_dict

def construct_prompt_dict(before_obj, object_desc, before_level, level, before_face, face, order):
    prompt_dict = {}
    prompt_dict["before_obj"] = before_obj
    prompt_dict["object"] = object_desc
    prompt_dict["before_face"] = before_face
    prompt_dict["face"] = face
    prompt_dict["before_level"] = before_level
    prompt_dict["level"] = level
    prompt_dict["order"] = order
    return prompt_dict

def frame_template(class_name, face, level, obj_props):

    class_name = class_name.split("_")[1:]
    object_prompt = ""

    for i in range(len(class_name)):
        object_prompt += class_name[i] + " "

    level_prompt = None

    if level == 1:
        level_prompt = "middle shelf"
    elif level == 2:
        level_prompt = "top shelf"
    elif level == 3:
        level_prompt = "top of the shelf"

    basic_prompts = [
        construct_prompt_dict("Place the ", object_prompt, "on the ", level_prompt, " facing ", face, ['o', 'l', 'f']),
        construct_prompt_dict("Reorient the ", object_prompt, " and place it on the ", level_prompt, "to face ", face, ['o', 'f', 'l'])
    ]

    if obj_props is not None:
        if '_' in obj_props:
            obj_props = obj_props.split('_')
        else:
            obj_props = [obj_props]

        for prop in obj_props:
            for prompt in basic_prompts.copy():
                new_prompt = prompt.copy()
                new_prompt["object"] = prop + " object "
                basic_prompts.append(
                    new_prompt
                )

    return basic_prompts

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

    json_file = path.Path(
        f"logs/reorient_dynamic/{args.seed:08d}-{args.face}-{args.level}.json"
    )
    if args.nogui and json_file.exists():
        logger.info(f"Already json_file exists: {json_file}")
        return

    all_files = {}
    reorient_key = ["should_reorient", "should_not_reorient"]
    class_key = [2, 3, 5, 11, 12, 15]
    face_key = ["front", "back", "right", "left"]
    level_key = ["middle shelf", "top shelf", "top of the shelf"]

    for key in reorient_key:
        all_files[key] = {}
        for class_id in class_key:
            all_files[key][class_id] = {}
            for face in face_key:
                all_files[key][class_id][face] = {}
                for level in level_key:
                    all_files[key][class_id][face][level] = 0

    all_file_names = []

    for file_name in tqdm(os.listdir(DEFAULT_DATA_PATH)):
        if file_name.endswith(".pkl"):

            with open(DEFAULT_DATA_PATH / file_name, "rb") as f:
                sample_dict = pickle.load(f)

            class_id = sample_dict["env_obs"]["target_class_id"]
            prompt = sample_dict["prompts"][0]

            if class_id in class_key:
                if sample_dict["is_reorientation_needed"]:
                    all_file_names.append(file_name)
                    all_files["should_reorient"][class_id][prompt["face"]][prompt["level"]] += 1
                else:
                    all_files["should_not_reorient"][class_id][prompt["face"]][prompt["level"]] += 1

    print(all_files) # {'should_reorient': 1160, 'should_not_reorient': 876, 'middle shelf': 430, 'top shelf': 463, 'top of the shelf': 267, 'class': {11: 259, 2: 516, 3: 529, 12: 302, 15: 78, 5: 352}}
    print(len(all_file_names))
    # assert False

    chosen_file_names = random.sample(all_file_names, 10)

    successful_trials = np.zeros(10)

    for trial in range(10):
        try:
            with open(DEFAULT_DATA_PATH / chosen_file_names[trial], "rb") as f:
                sample_dict = pickle.load(f)
                print(sample_dict.keys())

            args.seed = sample_dict["seed"]

            env = Env(
                class_ids=[2, 3, 5, 11, 12, 15],
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

            t_start = time.time()

            (
                reorient_poses,
                pickable,
                target_grasp_poses,
            ) = get_goal_oriented_reorient_poses(env)

            # succesful_grasp_poses = [sample[0] for sample in sample_dict["succesful_grasp_poses"]]
            index = np.random.choice(len(sample_dict["succesful_grasp_poses"]))
            succesful_grasp_poses = [sample_dict["succesful_grasp_poses"][index][0]]


            if sample_dict["is_reorientation_needed"]:
                logger.error("Failed to plan direct pick-and-place for {}".format(env.fg_class_name))
                success = False
                is_reorientation_needed = True

                grasp_poses = sample_dict["reorient_data"]["grasp_poses"]
                reorient_poses = sample_dict["reorient_data"]["reorient_poses"]
                pickable = sample_dict["reorient_data"]["pickable"]

                # choose random
                index = np.random.choice(len(grasp_poses))
                grasp_poses = np.array([grasp_poses[index]])
                reorient_poses = np.array([reorient_poses[index]])
                pickable = np.array([pickable[index]])

                result, _ = plan_dynamic_reorient(
                    env, grasp_poses, reorient_poses, pickable
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
            else:
                result = _reorient.plan_place_eval(env, succesful_grasp_poses)
                is_reorientation_needed = False
                reorient_data_dict = {}
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

            successful_trials[trial] = success

        except Exception as e:
            print(e)
            continue

    print("success", sum(successful_trials))

if __name__ == "__main__":
    main()
