import torch
import random
import numpy as np
from torch.utils.data import Dataset

import lightning as L
L.seed_everything(1)


# Ensure that all operations are deterministic on GPU (if used) for
# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Custom_Dataset(Dataset):
    def __init__(
            self,
            data_dir,
            fold,
            num_files,
            shuffle_atoms=False,
            d_proj=False,
            aug_times=1):
        self.num_samples = num_files
        self.data_dir = data_dir
        self.fold = fold
        self.shuffle = shuffle_atoms
        self.d_proj = d_proj
        self.aug_times = aug_times
        # self.max_lst = [0] * 6
        # self.min_lst = [10000] * 6

    def random_walk_closest(self, structure):
        visited = []
        n = len(structure)
        current_index = random.randint(0, n - 1)

        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = structure.get_distance(i, j)
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d

        while len(visited) < n:
            while True:
                next_index = None
                d_min = 1e10
                visited.append(current_index)
                idx_lst = list(range(n))
                random.shuffle(idx_lst)
                for i in idx_lst:
                    if i not in visited:
                        d = distance_matrix[current_index][i]
                        if d < d_min:
                            d_min = d
                            next_index = i
                if next_index is not None:
                    current_index = next_index
                    # print(d_min)
                else:
                    break
            if len(visited) < n:
                print(visited)
                raise ValueError("random walk error")
        # print(visited)
        return visited

    def __getitem__(self, idx):
        if self.aug_times > 1:
            idx = int(idx / self.aug_times)
        file_idx = idx
        data_dict = torch.load(f"{self.data_dir}_{self.fold}_{file_idx}.pth")
        feature_latt = data_dict["features_latt"]
        feature_abs = data_dict["features_abs_lst"]
        feature_abs_sum = data_dict["features_abs_sum"]
        feature_abs_avg = data_dict["features_abs_avg"]
        feature_angle = data_dict["features_angle_lst"]
        feature_angle_sum = data_dict["features_angle_sum"]
        feature_angle_avg = data_dict["features_angle_avg"]
        structure = data_dict["structure"]
        paths = data_dict["paths"]
        labels = data_dict["labels"][0]

        if (self.shuffle):
            if len(paths) > 0:
                indices = paths[random.randint(0, len(paths)-1)]
            else:
                # indices = torch.randperm(feature_angle.size(0))
                indices = self.random_walk_closest(structure)
            feature_angle = feature_angle[indices]
            feature_abs = feature_abs[indices]

        features_atom = []
        # Loop over each pair of features in feature_angle and feature_abs
        for i in range(feature_angle.shape[0]):
            if self.d_proj == False:
                if i==feature_angle.shape[0]-1: # last one
                    voxel_d = torch.full(feature_latt.unsqueeze(0).shape, 0, dtype=torch.float32)
                else:
                    d = structure.get_distance(indices[i], indices[i+1])
                    voxel_d = torch.full(feature_latt.unsqueeze(0).shape, d, dtype=torch.float32)
            else:
                if i==feature_angle.shape[0]-1: # last one
                    d = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
                else:
                    d = torch.tensor((structure.sites[indices[i]].frac_coords - structure.sites[indices[i+1]].frac_coords), dtype=torch.float32)

                hkl_range = torch.arange(-4, 4)
                h, k, l = torch.meshgrid(hkl_range, hkl_range, hkl_range, indexing='ij')
                hkl_tensor = torch.stack((h, k, l), -1).float()
                # lengths = torch.norm(hkl_tensor, dim=-1, keepdim=True)
                # lengths[lengths==0] = 1

                dot_products = torch.abs(torch.sum(hkl_tensor * d, dim=-1, keepdim=True))

                voxel_d = dot_products.squeeze() * feature_latt

            # self.max_lst[0] = max(self.max_lst[0], torch.max(feature_angle[i]))
            # self.max_lst[1] = max(self.max_lst[1], torch.max(feature_abs[i]))
            # self.max_lst[2] = max(self.max_lst[2], torch.max(voxel_d.to(dtype=feature_latt.dtype)))
            # self.min_lst[0] = min(self.min_lst[0], torch.min(feature_angle[i]))
            # self.min_lst[1] = min(self.min_lst[1], torch.min(feature_abs[i]))
            # self.min_lst[2] = min(self.min_lst[2], torch.min(voxel_d.to(dtype=feature_latt.dtype)))

            features_atom.append(torch.cat([feature_angle[i].unsqueeze(
                0), feature_abs[i].unsqueeze(0), voxel_d.reshape(feature_latt.unsqueeze(0).shape).to(dtype=feature_latt.dtype)], dim=0))

        features_atom = torch.cat(features_atom)
        
        # self.max_lst[3] = max(self.max_lst[3], torch.max(feature_angle_sum))
        # self.max_lst[4] = max(self.max_lst[4], torch.max(feature_abs_sum))
        # self.max_lst[5] = max(self.max_lst[5], torch.max(feature_latt))
        # self.min_lst[3] = min(self.min_lst[3], torch.min(feature_angle_sum))
        # self.min_lst[4] = min(self.min_lst[4], torch.min(feature_abs_sum))
        # self.min_lst[5] = min(self.min_lst[5], torch.min(feature_latt))

        features_all = torch.cat([features_atom,
                                  feature_angle_sum.view(1,
                                                         feature_abs_sum.size(0),
                                                         feature_abs_sum.size(1),
                                                         feature_abs_sum.size(2)),
                                  feature_abs_sum.view(1,
                                                       feature_abs_sum.size(0),
                                                       feature_abs_sum.size(1),
                                                       feature_abs_sum.size(2)),
                                  feature_latt.view(1,
                                                    feature_latt.size(0),
                                                    feature_latt.size(1),
                                                    feature_latt.size(2)),
                                  feature_angle_avg.view(1,
                                                         feature_abs_sum.size(0),
                                                         feature_abs_sum.size(1),
                                                         feature_abs_sum.size(2)),
                                  feature_abs_avg.view(1,
                                                       feature_abs_sum.size(0),
                                                       feature_abs_sum.size(1),
                                                       feature_abs_sum.size(2)),
                                  feature_latt.view(1,
                                                    feature_latt.size(0),
                                                    feature_latt.size(1),
                                                    feature_latt.size(2)),

                                  ])
        # print("max:", self.max_lst)
        # print("min:", self.min_lst)
        features_all = features_all.to(dtype=torch.float32)
        return features_all, labels.to(dtype=torch.float32)

    def __len__(self):
        return self.num_samples

