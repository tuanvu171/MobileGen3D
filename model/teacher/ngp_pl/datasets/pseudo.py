from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .ray_utils import (
    get_ray_directions,
    get_rays,
    rand_360_poses,
    rand_FF_poses
)

class PseudoDataset(Dataset):
    
    def __init__(
            self,
            W,
            H,
            K, 
            mean_radius, 
            min_radius, 
            max_radius,
            min_theta,
            max_theta,
            sr_downscale=8, 
            n_pseudo_data=10000, 
            ff=False,
            centered_poses=None,
            poses_train=None
    ):

        if ff and (centered_poses.any()) is None:
            raise ValueError('for ff dataset, centered poses is needed.')

        self.n_pseudo_data = n_pseudo_data
        self.ff = ff
        self.centered_poses = centered_poses
        self.H = H
        self.W = W
        self.K = K
        self.od_H = H // sr_downscale
        self.od_W = W // sr_downscale
        self.K_downscaled = K / sr_downscale
        self.mean_radius = mean_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.direction = get_ray_directions(self.H, self.W, self.K)
        self.sr_direction = get_ray_directions(self.od_H, self.od_W, self.K_downscaled)

        # # self.poses_train_interpolated = interpolate_poses(poses_train, n_pseudo_data, 4)
        # n_upsample = 10
        # for i in range(n_upsample):
        #     poses_train = upsample_poses(poses_train, k=5)
        self.poses_train = interpolate_poses(poses_train, n_pseudo_data, k=4)

        
    def __len__(self):
        return self.n_pseudo_data
    
    def __getitem__(self, idx):
        # get random poses
        if self.ff:
            c2w = rand_FF_poses(self.centered_poses)
        else:
            c2w = rand_360_poses(
                radius=[self.min_radius, self.max_radius],
                theta_range=[self.min_theta , self.max_theta]
            )
        
        # c2w = self.poses_train_interpolated[idx]
        c2w = self.poses_train[idx]
        
        rays_o, rays_d = get_rays(self.direction, c2w.clone())
        return {'rays_o': rays_o,
                'rays_d': rays_d,
                'pose': c2w
                # 'pose': c2w[:3, 3] if not self.ff else c2w[:3, :4]
                }
    
def compute_coverage(poses, k=4):
    num_poses = poses.shape[0]
    poses_flatenned = poses.T.reshape(num_poses, 3*4)
    nn = NearestNeighbors(n_neighbors=k+1).fit(poses_flatenned)
    distances, indices = nn.kneighbors(poses_flatenned)
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    coverage_values = np.zeros(num_poses)
    for i in range(num_poses):
        coverage_values[i] = 1/np.mean(distances[i])

    coverage_values = (coverage_values - np.min(coverage_values))/(np.max(coverage_values)-np.min(coverage_values))
    return coverage_values, distances, indices

def interpolate_poses(poses_train, n_pseudo_data, k=4):
    poses = poses_train.numpy()
    num_poses = poses.shape[0]
    num_poses_remove = 5
    num_poses_upsample = num_poses

    while num_poses < n_pseudo_data:
        coverage_values, neighbor_distances, neighbor_indices = compute_coverage(poses, k)

        # # In the first iteration, remove some least covered poses
        # if(num_poses==poses_train.numpy().shape[0]):
        #     poses_remove_indices = np.argsort(coverage_values)[:num_poses_remove]
        #     mask = np.ones(poses.shape[0], dtype=bool)
        #     mask[poses_remove_indices] = False
        #     poses = poses[mask]

        for i in range(num_poses_upsample):
            # choose one center pose
            probabilities = np.exp(coverage_values) / np.sum(np.exp(coverage_values))
            chosen_pose_index = np.random.choice(len(coverage_values), p=probabilities)

            neighbor_weights = neighbor_distances[chosen_pose_index]/np.sum(neighbor_distances[chosen_pose_index])
            stacked_neighbors = np.stack(poses[neighbor_indices[chosen_pose_index]], axis=0)
            upsampled_pose_neighbor = np.sum(stacked_neighbors * neighbor_weights[:, None, None], axis=0)
            center_pose_weight = np.random.uniform(0.7, 1)
            upsampled_pose = poses[chosen_pose_index]*center_pose_weight + upsampled_pose_neighbor*(1-center_pose_weight)
            poses = np.concatenate((poses, upsampled_pose[np.newaxis, :, :]), axis=0)
        
        num_poses = poses.shape[0]

    np.random.shuffle(poses)
    return torch.FloatTensor(poses)
