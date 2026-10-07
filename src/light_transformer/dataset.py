from torch.utils.data import Dataset
import torch
import numpy as np
import torch.nn.functional as F


class SpatialDataset(Dataset):
    def __init__(
        self,
        scene_data_paths,
        query_feat_paths,
        query_field_paths,
        fixed_num_scene_points,
        fixed_num_query_points,
        randomize=False,
        normalize_normals=True
    ):
        super(SpatialDataset, self).__init__()
        self.scene_data_paths = scene_data_paths
        self.query_feat_paths = query_feat_paths
        self.query_field_paths = query_field_paths
        self.fixed_scene_points = fixed_num_scene_points
        self.fixed_query_points = fixed_num_query_points
        self.randomize = randomize
        self.DTYPE = torch.float32
        query_data_features = np.load(self.query_feat_paths[0], allow_pickle=False)
        self.TOTAL_POINTS = len(query_data_features)
        self.normalize_normals = normalize_normals

    def __len__(self):
        return int(len(self.scene_data_paths) * (self.TOTAL_POINTS / self.fixed_query_points))

    def __getitem__(self, idx):
        scene_idx = idx % len(self.scene_data_paths)
        # print(f"loading {self.scene_data_paths[scene_idx]}")
        scene_data = np.load(self.scene_data_paths[scene_idx], allow_pickle=False)
        query_data_features = np.load(self.query_feat_paths[scene_idx], allow_pickle=False)
        query_data_values = np.load(
            self.query_field_paths[scene_idx], allow_pickle=False
        ).reshape(-1, 3)

        # if len(scene_data) > self.fixed_scene_points:
        #     index = np.random.choice(scene_data.shape[0], self.fixed_scene_points, replace=False)
        #     scene_data = scene_data[index]
        # if len(scene_data) < self.fixed_scene_points:
        #     index = np.random.choice(scene_data.shape[0], self.fixed_scene_points - len(scene_data), replace=True)
        #     scene_data = np.concatenate([scene_data, scene_data[index]], axis=0)
        
        # assert len(scene_data) == self.fixed_scene_points, f"scene_data should have {self.fixed_scene_points} points, but got {len(scene_data)}"

        if self.randomize:
            indices = torch.randperm(len(query_data_features))[: self.fixed_query_points]
            query_data_features = query_data_features[indices]
            query_data_values = query_data_values[indices]
        else:
            start_idx = idx // len(self.scene_data_paths)
            query_data_features = query_data_features[int(start_idx * self.fixed_query_points): int(start_idx * self.fixed_query_points) + self.fixed_query_points]
            query_data_values = query_data_values[int(start_idx * self.fixed_query_points): int(start_idx * self.fixed_query_points) + self.fixed_query_points]


        # # print(f"scene_data.shape: {scene_data.shape}, query_data_features.shape: {query_data_features.shape}, query_data_values.shape: {query_data_values.shape}")
        # if len(scene_data) > self.fixed_scene_points:
        #     # indices = torch.randperm(len(scene_data))[:self.fixed_scene_points]
        #     # scene_data = scene_data[indices] # THIS IS WRONG!!!! can never randomly select a subset !!!!
        #     pass
        # elif len(scene_data) < self.fixed_scene_points:
        #     # scene_data = np.concatenate([scene_data, scene_data[:self.fixed_scene_points - len(scene_data)]], axis=0)
        #     pass

        # if (
        #     len(query_data_features) > self.fixed_query_points
        # ):  # better to shuffle no matter what
        #     indices = torch.randperm(len(query_data_features))[: self.fixed_query_points]
        #     query_data_features = query_data_features[indices]
        #     query_data_values = query_data_values[indices]
        # # if np.sum(query_data_values) == 0:
        # #     print(f"query_data_values is zero for {scene_idx}")
        # #     return None
        scene_data = torch.tensor(scene_data, dtype=self.DTYPE)
        query_data_features = torch.tensor(query_data_features, dtype=self.DTYPE)
        query_data_values = torch.tensor(query_data_values, dtype=self.DTYPE)
        
        # light_mask = scene_data[..., 10] > 0.0
        # scene_data = scene_data[light_mask, 2] += 0.1
        # test_nan(scene_data,"scene_data")
        # test_nan(query_data_features, "query_data_features")
        # test_nan(query_data_values, "query_data_values")
        scene_data[torch.isnan(scene_data)] = 0.0
        query_data_features[torch.isnan(query_data_features)] = 0.0
        query_data_values[torch.isnan(query_data_values)] = 0.0
        # normalize the normals
        if self.normalize_normals:
            scene_data[..., 3:6] = F.normalize(scene_data[..., 3:6], dim=-1)
            query_data_features[..., 3:6] = F.normalize(query_data_features[..., 3:6], dim=-1)
        return scene_data, query_data_features, query_data_values


import random
class SpatialDataset_chunks(Dataset):
    def __init__(self, 
                 scene_data_paths, 
                 query_feat_paths, 
                 query_field_paths, 
                 fixed_num_scene_points, 
                 fixed_num_query_points,
                 randomize=False,
                normalize_normals=True):
        super(SpatialDataset_chunks, self).__init__()
        assert len(query_feat_paths) == len(query_field_paths), "Mismatch between feature and field file counts!"
        
        self.scene_data_paths = scene_data_paths  # Keeping for API consistency
        self.query_feat_paths = query_feat_paths  # Base paths (before picking chunks)
        self.query_field_paths = query_field_paths  # Base paths (before picking chunks)
        self.fixed_scene_points = fixed_num_scene_points  # Unused but kept for compatibility
        self.fixed_query_points = fixed_num_query_points
        self.DTYPE = torch.float32
        
        self.randomize = randomize
        self.normalize_normals = normalize_normals
        # query_data_features = np.load(self.query_feat_paths[0], allow_pickle=True)
        self.TOTAL_POINTS = 256*256*32
        assert self.fixed_query_points <= 16384, "fixed_query_points must not exceed the points in one chunk"
    def __len__(self):
        """ Returns the total number of base files (not the chunks). """
        return int(len(self.scene_data_paths) * (self.TOTAL_POINTS / self.fixed_query_points))

    
    def __getitem__(self, idx):
        """ Randomly selects a chunked file, loads it, and returns processed tensors. """
        scene_idx = idx % len(self.scene_data_paths)
        try:
            # Randomly pick a chunk index between 0 and 127
            chunk_idx = random.randint(0, 127)
            scene_data = np.load(self.scene_data_paths[scene_idx], allow_pickle=False)
            # Construct new file paths with the chunk index
            feat_chunk_path = self.query_feat_paths[scene_idx].replace(".npy", f"_{chunk_idx}.npy")
            field_chunk_path = self.query_field_paths[scene_idx].replace(".npy", f"_{chunk_idx}.npy")

            # Load chunked feature and field data
            query_data_features = np.load(feat_chunk_path, allow_pickle=False)
            query_data_values = np.load(field_chunk_path, allow_pickle=False).reshape(-1, 3)

            if self.randomize:
                indices = torch.randperm(len(query_data_features))[:self.fixed_query_points]
                query_data_features = query_data_features[indices]
                query_data_values = query_data_values[indices]

            scene_data = torch.tensor(scene_data, dtype=self.DTYPE)
            query_data_features = torch.tensor(query_data_features, dtype=self.DTYPE)
            query_data_values = torch.tensor(query_data_values, dtype=self.DTYPE)

            # Handle NaNs
            scene_data[torch.isnan(scene_data)] = 0.0
            query_data_features[torch.isnan(query_data_features)] = 0.0
            query_data_values[torch.isnan(query_data_values)] = 0.0

            # Normalize normals if present (assuming normal is in columns 3:6)
            if self.normalize_normals:
                scene_data[..., 3:6] = F.normalize(scene_data[..., 3:6], dim=-1)
                query_data_features[..., 3:6] = F.normalize(query_data_features[..., 3:6], dim=-1)

            return scene_data, query_data_features, query_data_values

        except Exception as e:
            print(f"Exception in __getitem__ for index {scene_idx}: {e}", flush=True)
            raise
