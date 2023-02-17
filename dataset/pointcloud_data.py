import os
import path
import pickle
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

home = path.Path("/hddscratch/umishra31/").expanduser()
DEFAULT_DATA_PATH = home / "graspingbot" / "reorient_pointcloud"

class PointCloudDataset(torch.utils.data.Dataset):
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

            # if len(self.all_data) > 5000:
            #     break

        self.class_wise_bounds = self.prepare_class_wise_bounds()

        for i in range(len(self.all_data)):
            chosen_point_cloud, orientation, object_onehot = self.all_data[i]

            class_id = np.argmax(object_onehot)
            bounds = self.class_wise_bounds[class_id]

            modified_point_cloud = chosen_point_cloud/bounds

            assert np.all(np.abs(modified_point_cloud) <= 1)

            self.all_data[i] = (modified_point_cloud, np.array(orientation), object_onehot)

        print('Loaded data...', len(self.all_data))

        # self.print_stats()


    def __len__(self):
        return len(self.all_data)

    def print_stats(self):

        object_wise_dict = {}

        for data in self.all_data:
            _, orientation, object_onehot = data

            class_id = np.argmax(object_onehot)

            if class_id not in object_wise_dict:
                object_wise_dict[class_id] = {}

            orientation = tuple(orientation)

            if orientation not in object_wise_dict[class_id]:
                object_wise_dict[class_id][orientation] = 0

            object_wise_dict[class_id][orientation] += 1

        for class_id in object_wise_dict:
            print('Class:', self.object_class_ids[class_id])
            for orientation in object_wise_dict[class_id]:
                print('\t', orientation, object_wise_dict[class_id][orientation])

    def visualize_ten_random(self):

        import open3d as o3d

        pcds = []

        for i in range(10):
            chosen_point_cloud, orientation, object_onehot = self.all_data[np.random.randint(0, len(self.all_data))]

            parsed_point_cloud = self.parse_point_cloud(chosen_point_cloud, np.argmax(object_onehot))

            pcds.append(parsed_point_cloud + i*2)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(pcds, axis=0))
        o3d.visualization.draw_geometries([pcd])

    def parse_file(self, file_path):
        
        # load data

        file_data = []

        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            pcd_in_obj = data['pcd_in_obj']
            pcd_in_obj_full = data['pcd_in_obj_full']
            class_id = data['class_id']
            pose = data['pose']

            if pcd_in_obj.shape[0] == 0 or pcd_in_obj_full.shape[0] == 0:
                return file_data

            # get object onehot

            object_onehot = np.zeros(self.num_object_ids)
            object_onehot[self.object_class_ids.index(class_id)] = 1

            # get orientation

            orientation = pose[1]

            if pcd_in_obj.shape[0] < 3000:
                replace_flag = True
                num_iterations = 1
            else:
                replace_flag = False
                num_iterations = 5

            for _ in range(num_iterations):
                chosen_point_cloud = pcd_in_obj[np.random.choice(pcd_in_obj.shape[0], 3000, replace=replace_flag)]

                file_data.append((chosen_point_cloud, orientation, object_onehot))

            orientation = np.zeros_like(orientation)

            if pcd_in_obj_full.shape[0] < 3000:
                replace_flag = True
            else:
                replace_flag = False

            for _ in range(1):
                chosen_point_cloud = pcd_in_obj_full[np.random.choice(pcd_in_obj_full.shape[0], 3000, replace=replace_flag)]

                file_data.append((chosen_point_cloud, orientation, object_onehot))

        except Exception as e:
            print(e)
            pass

        return file_data

    def prepare_class_wise_bounds(self):

        class_wise_bounds = {}

        for data in self.all_data:
            chosen_point_cloud, orientation, object_onehot = data

            class_id = np.argmax(object_onehot)

            if class_id not in class_wise_bounds:
                class_wise_bounds[class_id] = []

            class_wise_bounds[class_id].append(np.max(np.abs(chosen_point_cloud), axis=0))

        for class_id in class_wise_bounds:
            class_wise_bounds[class_id] = np.max(np.array(class_wise_bounds[class_id]), axis=0)

        return class_wise_bounds

    def __getitem__(self, idx):
        chosen_point_cloud, orientation, object_onehot = self.all_data[idx]

        return chosen_point_cloud, orientation, object_onehot

    def parse_point_cloud(self, point_cloud, class_id):

        bounds = self.class_wise_bounds[class_id]

        # multiple by bounds axis wise

        modified_point_cloud = point_cloud * bounds

        return modified_point_cloud

def get_data_loader(batch_size=128, num_workers=4, train_test_split=0.9, data_path=DEFAULT_DATA_PATH, device='cuda'):
    
    dataset = PointCloudDataset(data_path=data_path, device=device)

    train_size = int(train_test_split * len(dataset))
    test_size = len(dataset) - train_size 

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader, dataset

def main():
    train_loader, test_loader, dataset = get_data_loader()

    print(len(train_loader), len(test_loader), len(dataset))

    # dataset.visualize_ten_random()

    
if __name__ == '__main__':
    main()