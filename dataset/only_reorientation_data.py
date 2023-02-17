import os
import path
import pickle
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

home = path.Path("/hddscratch/umishra31/").expanduser()
DEFAULT_DATA_PATH = home / 'graspingbot' / 'reorient_dynamic2'

class ReorientationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path=DEFAULT_DATA_PATH, device='cuda', selected_class_id=None, selected_face=None):
        
        self.data_path = data_path
        self.device = device

        assert os.path.exists(data_path)

        self.object_class_ids = [2, 3, 5, 11, 12, 15]
        self.num_object_ids = len(self.object_class_ids)

        self.selected_class_id = selected_class_id
        self.selected_face = selected_face

        self.file_paths = []
        self.all_data = []

        print('Loading data...', data_path, len(os.listdir(data_path)))

        for file in tqdm(os.listdir(data_path)):
            if file.endswith('.pkl'):
                self.file_paths.append(os.path.join(data_path, file))
                self.all_data.extend(self.parse_file(os.path.join(data_path, file)))

            if len(self.all_data) > 5000:
                break

        self.all_pose_data = []
        self.all_grasp_data = []
        self.all_reorient_data = []

        self.min_reorientations = 100000
        self.min_grasps = 100000

        for data in self.all_data:
            if data[6].shape[0] < self.min_reorientations:
                self.min_reorientations = data[6].shape[0]
            if data[5].shape[0] < self.min_grasps:
                self.min_grasps = data[5].shape[0]

            if len(data[3]) > 0:
                self.all_pose_data.extend([data[3]])
            self.all_grasp_data.extend(data[5])
            self.all_reorient_data.extend(data[6])

        self.all_pose_data = np.array(self.all_pose_data)
        self.all_grasp_data = np.array(self.all_grasp_data)
        self.all_reorient_data = np.array(self.all_reorient_data)

        self.all_pose_data = self.all_pose_data.reshape(-1, self.all_pose_data.shape[-1])
        self.all_grasp_data = self.all_grasp_data.reshape(-1, self.all_grasp_data.shape[-1])
        self.all_reorient_data = self.all_reorient_data.reshape(-1, self.all_reorient_data.shape[-1])

        self.pose_mean = np.mean(self.all_pose_data, axis=0)
        self.pose_std = np.std(self.all_pose_data, axis=0)
        self.min_pose_std = min(self.pose_std)

        self.grasp_mean = np.mean(self.all_grasp_data, axis=0)
        self.grasp_std = np.std(self.all_grasp_data, axis=0)
        self.min_grasp_std = min(self.grasp_std)

        self.reorient_mean = np.mean(self.all_reorient_data, axis=0)
        self.reorient_std = np.std(self.all_reorient_data, axis=0)
        self.min_reorient_std = min(self.reorient_std)

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

            text_prompt = data['prompts']
            target_grasp_poses = np.array([grasp[0] for grasp in data['succesful_grasp_poses']])

            total_grasps = len(target_grasp_poses)

            target_class_id = data["env_obs"]["target_class_id"]

            if self.selected_class_id is not None and target_class_id != self.selected_class_id:
                print('skipping class', target_class_id, self.selected_class_id)
                return []

            object_onehot = np.zeros(self.num_object_ids)
            object_onehot[self.object_class_ids.index(target_class_id)] = 1

            pose = np.hstack(data["env_obs"]['place_pose'])
            grasp_poses = np.array(data['reorient_data']['grasp_poses'])
            reorient_poses = np.array(data['reorient_data']['reorient_poses'])
            reorientable_pred = np.array(data['reorient_data']['reorientable_pred'])
            trajectory_lengths = np.array(data['reorient_data']['trajectory_length_pred'])

            if grasp_poses.shape[0] < 8 or target_grasp_poses.shape[0] < 8:
                return []
            
            # replace nan with 0
            trajectory_lengths[np.isnan(trajectory_lengths)] = 0

            reorientation_prediction = np.concatenate([grasp_poses, reorient_poses, trajectory_lengths[:, None]], axis=1)

            rgb_image = data['env_obs']['rgb']/255.0
            heightmap = data['env_obs']['depth']
            heightmap = heightmap/heightmap.max()
            segmentation = data['env_obs']['fg_mask']  
            segmmap = data['env_obs']['segmmap']
            pointmap = data['env_obs']['pointmap']  
            rgbd_image = np.concatenate([rgb_image, heightmap[:, :, None]], axis=2)
            all_segmentation = []
            all_class_ids = self.object_class_ids
            for segm in data["env_obs"]["all_segms"]:
                if segm[1] in all_class_ids:
                    all_segmentation.append(segm[0])

            all_segmentation = np.array(all_segmentation)      
            
            rgb_image = rgb_image.transpose(2, 0, 1)
            rgbd_image = rgbd_image.transpose(2, 0, 1)

            pile_file = data['env_obs']['pile_file']

            for prompt in text_prompt:
                if self.selected_face is not None and self.selected_face not in prompt['face']:
                    continue
                file_data.append((rgb_image, self.prompt2str(prompt), rgbd_image, pose, object_onehot, target_grasp_poses, reorientation_prediction, segmentation, pile_file))

        except Exception as e:
            print(e)
            pass

        return file_data

    def __getitem__(self, idx):
        image, text, rgbd, pose, object_onehot, target_grasps, reorientation_prediction, segmentation, pile_file = self.all_data[idx]

        image = torch.Tensor(image) #.to(self.device)
        pose = torch.Tensor((pose - self.pose_mean) / (self.pose_std + self.min_pose_std + 0.001)) #.to(self.device)
        rgbd = torch.Tensor(rgbd) #.to(self.device)
        object_onehot = torch.Tensor(object_onehot) #.to(self.device)
        target_grasps = torch.Tensor((target_grasps - self.grasp_mean[None, :]) / (self.grasp_std[None, :] + self.min_grasp_std + 0.001)) #.to(self.device)
        reorientation_prediction = torch.Tensor((reorientation_prediction - self.reorient_mean[None, :]) / (self.reorient_std[None, :] + self.min_reorient_std + 0.001)) #.to(self.device)
        segmentation = torch.Tensor(segmentation) #.to(self.device)
        # print(image.shape, text, rgbd.shape, pose.shape, object_onehot.shape, target_grasps.shape, reorientation_prediction.shape)

        # choose self.min_reorientations grasps
        index_gra = np.random.choice(target_grasps.shape[0], self.min_grasps, replace=False)
        index_reo = np.random.choice(reorientation_prediction.shape[0], self.min_reorientations, replace=False)
        
        return image, text, pose, rgbd, object_onehot, target_grasps[index_gra, :], reorientation_prediction[index_reo, :], segmentation, pile_file

    def parse_prediction(self, pose=None, target_grasps=None, reorientation_pred=None, object_onehot=None):

        if pose is not None:
            pose = pose * (self.pose_std + self.min_pose_std + 0.001) + self.pose_mean
        
        if target_grasps is not None:
            target_grasps = target_grasps * (self.grasp_std[None, :] + self.min_grasp_std + 0.001) + self.grasp_mean[None, :]

        if reorientation_pred is not None:
            reorientation_pred = reorientation_pred * (self.reorient_std[None, :] + self.min_reorient_std + 0.001) + self.reorient_mean[None, :]

        if object_onehot is not None:
            object_onehot = self.object_class_ids[np.argmax(object_onehot)]

        return pose, target_grasps, reorientation_pred, object_onehot

def get_data_loader(batch_size=128, num_workers=4, train_test_split=0.9, data_path=DEFAULT_DATA_PATH, selected_class_id=None, selected_face=None, device='cuda'):
    
    dataset = ReorientationDataset(data_path=data_path, device=device, selected_class_id=selected_class_id, selected_face=selected_face)

    print("######################################################################")
    print("Loaded dataset consisting of {} samples".format(len(dataset))) 
    print("######################################################################")

    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader, dataset
