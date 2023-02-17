import os
import path
import pickle
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

home = path.Path("/ssdscratch/umishra31/").expanduser()
DEFAULT_DATA_PATH = home / 'reorient_data' / 'reorient_dynamic'

class ReorientationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=DEFAULT_DATA_PATH, device='cuda'):
        
        self.data_path = data_path
        self.device = device

        assert os.path.exists(data_path)

        self.object_class_ids = [2, 3, 5, 11, 12, 15, 16]
        self.num_object_ids = len(self.object_class_ids)

        self.file_paths = []
        self.all_data = []

        print('Loading data...', data_path, len(os.listdir(data_path)))

        for file in tqdm(os.listdir(data_path)):
            if file.endswith('.pkl'):
                self.file_paths.append(os.path.join(data_path, file))
                self.all_data.extend(self.parse_file(os.path.join(data_path, file)))

            if len(self.all_data) > 5000:
                break

        self.all_pose_data = np.array([data[2] for data in self.all_data])
        self.all_grasp_data = np.array([data[4] for data in self.all_data])
        self.all_reorient_data = np.array([data[5] for data in self.all_data])

        self.all_grasp_data = self.all_grasp_data.reshape(-1, self.all_grasp_data.shape[-1])
        self.all_reorient_data = self.all_reorient_data.reshape(-1, self.all_reorient_data.shape[-1])

        self.pose_mean = np.mean(self.all_pose_data, axis=0)
        self.pose_std = np.std(self.all_pose_data, axis=0)

        self.grasp_mean = np.mean(self.all_grasp_data, axis=0)
        self.grasp_std = np.std(self.all_grasp_data, axis=0)

        self.reorient_mean = np.mean(self.all_reorient_data, axis=0)
        self.reorient_std = np.std(self.all_reorient_data, axis=0)

    def __len__(self):
        return len(self.all_data)

    def prompt2str(self, prompt):
        order = prompt['order']

        text = ''

        for i in order:
            if i == 'o':
                text += prompt['before_obj'] + prompt['object']
            elif i == 'f':
                text += prompt['before_face'] + prompt['face']
            elif i == 'l':
                text += prompt['before_level'] + prompt['level']
            else:
                assert False

        return text

    def parse_file(self, file_path):
        
        # load data

        file_data = []

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            text_prompt = data['prompt']
            target_grasp_poses = data['target_grasp_poses']

            total_grasps = len(target_grasp_poses)

            class_ids = data['dynamic_data']['env_obs']['class_ids']
            object_ids = data['dynamic_data']['env_obs']['object_ids']
            target_object_id = data['dynamic_data']['pile_info']['object_id']

            assert len(class_ids) == len(object_ids)
            assert data['dynamic_data']['pile_info']['object_id'] == data['dynamic_data']['env_obs']['fg_object_id']

            target_class_id = class_ids[object_ids.index(target_object_id)]
            object_onehot = np.zeros(self.num_object_ids)
            object_onehot[self.object_class_ids.index(target_class_id)] = 1

            pose = np.hstack(data['place_pose'])
            grasp_poses = data['dynamic_data']['grasp_poses']
            reorient_poses = data['dynamic_data']['reorient_poses']
            reorientable_pred = data['dynamic_data']['reorientable_pred']
            trajectory_length_pred = data['dynamic_data']['trajectory_length_pred']

            N_grasp, N_reorient = reorient_poses.shape[0], reorient_poses.shape[1]

            N_top = 3
            i_grasp = np.arange(N_grasp)[:, None].repeat(N_top, axis=1)
            i_reorient = np.argsort(reorientable_pred, axis=1)[:, -N_top:]

            reorientable_pred = reorientable_pred[i_grasp, i_reorient]
            trajectory_length_pred = trajectory_length_pred[i_grasp, i_reorient]
            grasp_poses = grasp_poses[i_grasp, i_reorient]
            reorient_poses = reorient_poses[i_grasp, i_reorient]

            reorientable_pred = reorientable_pred.reshape(N_grasp * N_top)
            trajectory_length_pred = trajectory_length_pred.reshape(N_grasp * N_top)
            grasp_poses = grasp_poses.reshape(N_grasp * N_top, 7)
            reorient_poses = reorient_poses.reshape(N_grasp * N_top, 7)

            indices1 = np.argsort(trajectory_length_pred)
            indices2 = np.argsort(reorientable_pred)[::-1]
            indices = np.r_[indices1[:3], indices2[:3], indices1[3:12], indices2[3:12]]

            grasp_poses = grasp_poses[indices]
            reorient_poses = reorient_poses[indices]
            trajectory_lengths = trajectory_length_pred[indices]

            # replace nan with 0
            trajectory_lengths[np.isnan(trajectory_lengths)] = 0

            reorientation_prediction = np.concatenate([reorient_poses, trajectory_lengths[:, None]], axis=1)

            rgb_image = data['dynamic_data']['env_obs']['rgb']
            heightmap = data['dynamic_data']['env_obs']['depth']
            segmentation = data['dynamic_data']['env_obs']['fg_mask']            

            image = rgb_image #np.concatenate([rgb_image, heightmap[:, :, None]], axis=2) #, segmentation[:, :, None]], axis=2)
            image = image.transpose(2, 0, 1) #[:, None, :, :]

            for prompt in text_prompt:
                chosen_grasps = np.random.choice(total_grasps, 8, replace=False)
                chosen_grasp_poses = np.array([target_grasp_poses[i] for i in chosen_grasps])
                file_data.append((image, self.prompt2str(prompt), pose, object_onehot, chosen_grasp_poses, reorientation_prediction))

        except Exception as e:
            print(e)
            pass

        return file_data

    def __getitem__(self, idx):
        image, text, pose, object_onehot, grasps, reorientation_prediction = self.all_data[idx]

        image = torch.Tensor(image) #.to(self.device)
        pose = torch.Tensor((pose - self.pose_mean) / (self.pose_std+0.001)) #.to(self.device)
        object_onehot = torch.Tensor(object_onehot) #.to(self.device)
        grasps = torch.Tensor((grasps - self.grasp_mean[None, :]) / (self.grasp_std[None, :]+0.001)) #.to(self.device)
        reorientation_prediction = torch.Tensor((reorientation_prediction - self.reorient_mean[None, :]) / (self.reorient_std[None, :]+0.001)) #.to(self.device)

        return image, text, pose, object_onehot, grasps, reorientation_prediction

def get_data_loader(batch_size=128, num_workers=4, train_test_split=[0.8, 0.1], data_path=DEFAULT_DATA_PATH, device='cuda'):
    
    dataset = ReorientationDataset(data_path=data_path, device=device)

    train_size = int(train_test_split[0] * len(dataset))
    val_size = int(train_test_split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset
