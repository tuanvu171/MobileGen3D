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

        # self.poses_train_interpolated = interpolate_poses(poses_train, n_pseudo_data, 4)
        n_upsample = 10
        for i in range(n_upsample):
            poses_train = upsample_poses(poses_train, k=5)

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
                'pose': c2w[:3, 3] if not self.ff else c2w[:3, :4]
                }
    
# def interpolate_poses(poses_train, M):
#     """
#     Interpolate poses in 'poses_train' to generate 'poses_train_interpolated' with 'M' poses.

#     Args:
#     poses_train (numpy.ndarray): An array of size (N, 3, 4) containing N poses in OpenGL convention.
#     M (int): The desired number of interpolated poses.

#     Returns:
#     numpy.ndarray: An array of size (M, 3, 4) containing interpolated poses.
#     """
#     N = poses_train.shape[0]
#     interpolation_steps = (M - 1) // (N - 1)
    
#     poses_train_interpolated = np.zeros((M, 3, 4), dtype=np.float32)
    
#     for i in range(N - 1):
#         pose_start = poses_train[i]
#         pose_end = poses_train[i + 1]
        
#         for step in range(interpolation_steps + 1):
#             alpha = step / interpolation_steps
#             interpolated_pose = pose_start + alpha * (pose_end - pose_start)
#             poses_train_interpolated[i * interpolation_steps + step] = interpolated_pose
    
#     # Copy the last pose from 'poses_train' to 'poses_train_interpolated'
#     poses_train_interpolated[-1] = poses_train[-1]
    
#     return torch.FloatTensor(poses_train_interpolated)

def upsample_poses(poses_train, k):
    poses_train = poses_train.numpy()
    N = poses_train.shape[0]
    coverage_values = np.zeros(N)

    pose_fatenned = poses_train[:, :3, :3].reshape(-1, 3*3)  # Reshape to 2D
    nn = NearestNeighbors(n_neighbors=k+1).fit(pose_fatenned) # k+1 because one of the neighbors is itself
    distances, indices = nn.kneighbors(pose_fatenned)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    # Compute coverage values
    for i in range(N):
        coverage_values[i] = 1/np.mean(distances[i])

    # normalize to 0-1
    coverage_values = (coverage_values - np.min(coverage_values))/(np.max(coverage_values)-np.min(coverage_values))

    # the lower coverage value, the higher chance for upsampling
    rand_vect = np.random.rand(N)
    is_upsample = coverage_values <= rand_vect

    for i in range(N):
        if(is_upsample[i]):
            neighbor_weights = distances[i]/np.sum(distances[i])
            stacked_neighbors = np.stack(poses_train[indices[i]], axis=0)
            upsampled_pose = np.sum(stacked_neighbors * neighbor_weights[:, None, None], axis=0)
            poses_train = np.concatenate((poses_train, upsampled_pose[np.newaxis, :, :]), axis=0)

    return torch.FloatTensor(poses_train)
        
def interpolate_poses(poses_train, M, k=4):
    poses_train = poses_train.numpy()
    N = poses_train.shape[0]
    N_interpolate = -(-M//N) - 1

    pose_fatenned = poses_train[:, :3, :3].reshape(-1, 3*3)  # Reshape to 2D
    nn = NearestNeighbors(n_neighbors=k).fit(pose_fatenned) # k+1 because one of the neighbors is itself
    _, indices = nn.kneighbors(pose_fatenned)

    # Generate "N_interpolate" more poses for each original pose
    for i in range(N):
        indice_neighbors = indices[i]
        weights = np.random.rand(N_interpolate, k)
        stacked_neighbors = np.stack(poses_train[indice_neighbors], axis=0)
        for j in range(N_interpolate):
            # The interpolated pose is the weighted average of "k" neighbor poses
            weight = weights[j]
            weight = weight/np.sum(weight)
            interpolated_pose = np.sum(stacked_neighbors * weight[:, None, None], axis=0)
            poses_train = np.concatenate((poses_train, interpolated_pose[np.newaxis, :, :]), axis=0)
    poses_train = poses_train[np.random.permutation(poses_train.shape[0])]
    return torch.FloatTensor(poses_train)