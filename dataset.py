import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class EMT_SCF_Dataset(Dataset):
    def __init__(self, h5_path, sample_indices=None, split="train", mode="heat_only",
                 num_p2_points=2000, num_p3_points=4000, excite_per_sample=16, base_seed=42):
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.mode = mode
        self.num_p2_points = num_p2_points
        self.num_p3_points = num_p3_points
        self.excite_per_sample = excite_per_sample
        self.base_seed = base_seed
        self.current_epoch = 0

        self.hf = None

        with h5py.File(h5_path, 'r') as hf_temp:
            total_h5_samples = hf_temp['v_pair'].shape[0]
            self.total_points = hf_temp['coords_3d_field_norm'].shape[0]
            self.coords_3d_field_norm = torch.from_numpy(hf_temp['coords_3d_field_norm'][:])
            self.labels_norm = torch.from_numpy(hf_temp['labels_norm'][:])

        if sample_indices is None:
            self.sample_indices = np.arange(total_h5_samples)
        else:
            self.sample_indices = np.array(sample_indices)

        self.num_samples = len(self.sample_indices)
        self.total_items = self.num_samples * self.excite_per_sample

    def __len__(self):
        return self.total_items

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_physical_weights(self):
        """ 返回当前子集的物理样本采样权重，专供 PhysicalWeightedSampler 使用 """
        weights = []
        for sample_idx in range(self.num_samples):
            physical_id = self.sample_indices[sample_idx]
            label_norm = self.labels_norm[physical_id]

            cx, cy, w, l = label_norm.tolist()
            area = w * l
            dist_x = 1.0 - abs(cx) - w / 2.0
            dist_y = 1.0 - abs(cy) - l / 2.0
            dist_bound = min(dist_x, dist_y)

            is_small = area < 0.025
            is_edge = dist_bound < 0.15

            # 🚀 温和权重：防止大盘崩坏！
            weight = 1.0
            if is_small: weight += 0.3
            if is_edge: weight += 0.3

            weights.append(weight)
        return weights

    def _generate_stratified_2d_points(self, label_norm):
        if self.num_p2_points == 0:
            return torch.zeros((0, 2)), torch.zeros((0, 1))

        cx, cy, w, l = label_norm.tolist()
        x_min, x_max = cx - w / 2, cx + w / 2
        y_min, y_max = cy - l / 2, cy + l / 2

        N_inside = int(self.num_p2_points * 0.2)
        N_bound = int(self.num_p2_points * 0.3)
        N_global = self.num_p2_points - N_inside - N_bound

        pts_in_x = torch.rand(N_inside) * w + x_min
        pts_in_y = torch.rand(N_inside) * l + y_min
        pts_inside = torch.stack([pts_in_x, pts_in_y], dim=1)

        bound_x = torch.empty(N_bound);
        bound_y = torch.empty(N_bound)
        edge_choices = torch.randint(0, 4, (N_bound,))

        idx0 = edge_choices == 0
        bound_x[idx0] = torch.rand(idx0.sum()) * w + x_min;
        bound_y[idx0] = y_max
        idx1 = edge_choices == 1
        bound_x[idx1] = torch.rand(idx1.sum()) * w + x_min;
        bound_y[idx1] = y_min
        idx2 = edge_choices == 2
        bound_x[idx2] = x_min;
        bound_y[idx2] = torch.rand(idx2.sum()) * l + y_min
        idx3 = edge_choices == 3
        bound_x[idx3] = x_max;
        bound_y[idx3] = torch.rand(idx3.sum()) * l + y_min
        pts_bound = torch.stack([bound_x, bound_y], dim=1) + torch.randn(N_bound, 2) * 0.05

        pts_global = torch.rand((N_global, 2)) * 2.0 - 1.0
        coords_2d = torch.clamp(torch.cat([pts_inside, pts_bound, pts_global], dim=0), -1.0, 1.0)

        in_x = (coords_2d[:, 0] >= x_min) & (coords_2d[:, 0] <= x_max)
        in_y = (coords_2d[:, 1] >= y_min) & (coords_2d[:, 1] <= y_max)
        heat_target = (in_x & in_y).float().unsqueeze(1)

        perm = torch.randperm(self.num_p2_points)
        return coords_2d[perm], heat_target[perm]

    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')

        sample_idx = idx // self.excite_per_sample
        inner_idx = idx % self.excite_per_sample
        physical_id = self.sample_indices[sample_idx]

        if self.excite_per_sample == 1:
            excite_idx = 0
        elif self.excite_per_sample == 16:
            excite_idx = inner_idx
        else:
            excite_idx = ((self.current_epoch * self.excite_per_sample) + inner_idx) % 16

        v_pair = torch.from_numpy(self.hf['v_pair'][physical_id])
        label_norm = self.labels_norm[physical_id]
        coords_2d, heat_target = self._generate_stratified_2d_points(label_norm)

        batch = {
            "v_pair": v_pair,
            "coords_2d": coords_2d,
            "heat_target": heat_target,
            "excite_idx": torch.tensor(excite_idx, dtype=torch.long),
            "label_xywl": label_norm
        }

        if self.mode == "joint":
            if self.split == "val":
                g = torch.Generator(device='cpu')
                g.manual_seed(self.base_seed + int(physical_id) * 100 + int(excite_idx))
                point_indices = torch.randperm(self.total_points, generator=g)[:self.num_p3_points]
            else:
                point_indices = torch.randperm(self.total_points)[:self.num_p3_points]

            coords_3d_full = self.coords_3d_field_norm
            A_target_full = self.hf['A_delta_norm'][physical_id, excite_idx, :, :]
            A0_target_full = self.hf['A0_norm'][excite_idx, :, :]

            coords_3d = coords_3d_full[point_indices]
            A_target = torch.from_numpy(A_target_full)[point_indices]
            A0_target = torch.from_numpy(A0_target_full)[point_indices]

            batch["coords_3d"] = coords_3d
            batch["A_target"] = A_target
            batch["A0_target"] = A0_target

        return batch