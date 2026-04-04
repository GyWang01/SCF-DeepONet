import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class EMT_SCF_Dataset(Dataset):
    def __init__(self, h5_path, sample_indices=None, split="train", mode="heat_only",
                 num_p2_points=2000, num_p3_points=4000, num_pde_points=1024,
                 excite_per_sample=16, base_seed=42, defect_h=5.0):
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.mode = mode
        self.num_p2_points = num_p2_points
        self.num_p3_points = num_p3_points
        self.num_pde_points = num_pde_points  # 【新增】PDE配点数量
        self.excite_per_sample = excite_per_sample
        self.base_seed = base_seed
        self.current_epoch = 0

        # 【新增】缺陷的固定深度（单位需与你坐标系一致，假设为 5.0 毫米）
        # 这个值将用于后续构建 3D Soft Mask
        self.defect_h = defect_h

        self.hf = None

        with h5py.File(h5_path, 'r') as hf_temp:
            total_h5_samples = hf_temp['v_pair'].shape[0]
            self.total_points = hf_temp['coords_3d_field_norm'].shape[0]
            self.coords_3d_field_norm = torch.from_numpy(hf_temp['coords_3d_field_norm'][:])
            self.labels_norm = torch.from_numpy(hf_temp['labels_norm'][:])

            # 【新增】读取标度，用于 PDE 选点时的物理空间过滤
            self.max_field_xyz = torch.from_numpy(hf_temp.attrs['MAX_FIELD_XYZ'][:])

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

        bound_x = torch.empty(N_bound)
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

    def _sample_pde_points(self):
        """
        【新增】专门用于筛选 PDE 物理配点
        按照专家建议：纳入深层铁块、远场空气；避开线圈源区(Source Zone)
        """
        # 将归一化坐标还原为物理坐标用于逻辑判断
        z_real = self.coords_3d_field_norm[:, 2] * self.max_field_xyz[2]

        # 启发式过滤：假设线圈和间隙（Source Zone）在 z = [0, 5] 的区间内
        # 这里把 z < -2 (深层/核心铁块) 和 z > 15 (远场空气) 作为安全区
        # 具体数值你可以根据 COMSOL 建模的实际物理坐标域进行微调
        safe_mask = (z_real < -2.0) | (z_real > 15.0)
        safe_indices = torch.nonzero(safe_mask).squeeze()

        # 如果安全区点数不足（通常不可能），退化为全局随机采样
        if len(safe_indices) < self.num_pde_points:
            return torch.randperm(self.total_points)[:self.num_pde_points]

        # 在安全区内随机抽取指定数量的 PDE 配点
        perm = torch.randperm(len(safe_indices))[:self.num_pde_points]
        return safe_indices[perm]

    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, 'r')

        local_sample_idx = idx // self.excite_per_sample
        physical_id = self.sample_indices[local_sample_idx]

        if self.excite_per_sample == 16:
            excite_idx = idx % 16
        else:
            inner_idx = idx % self.excite_per_sample
            excite_idx = ((self.current_epoch * self.excite_per_sample) + inner_idx) % 16

        v_pair = torch.from_numpy(self.hf['v_pair'][physical_id])
        label_norm = self.labels_norm[physical_id]
        coords_2d, heat_target = self._generate_stratified_2d_points(label_norm)

        batch = {
            "v_pair": v_pair,
            "coords_2d": coords_2d,
            "heat_target": heat_target,
            "excite_idx": torch.tensor(excite_idx, dtype=torch.long),
            "label_xywl": label_norm,
            "defect_h": torch.tensor(self.defect_h, dtype=torch.float32)  # 【新增】透传几何先验
        }

        # 只要不是纯热图模式（即 joint 强监督或 PDE 模式），都加载 3D 场相关数据
        if self.mode in ["joint", "pde"]:
            coords_3d_full = self.coords_3d_field_norm
            A_target_full = self.hf['A_delta_norm'][physical_id, excite_idx, :, :]
            A0_target_full = self.hf['A0_norm'][excite_idx, :, :]

            # --- 1. 强监督点采样 (保持原逻辑) ---
            if self.mode == "joint" and self.split == "val":
                g = torch.Generator(device='cpu')
                g.manual_seed(self.base_seed + int(physical_id) * 100 + int(excite_idx))
                point_indices = torch.randperm(self.total_points, generator=g)[:self.num_p3_points]
            else:
                point_indices = torch.randperm(self.total_points)[:self.num_p3_points]

            batch["coords_3d"] = coords_3d_full[point_indices]
            batch["A_target"] = torch.from_numpy(A_target_full)[point_indices]
            # 此时 A0_target 可以留作强监督的辅助参考，也可以不返回
            batch["A0_target"] = torch.from_numpy(A0_target_full)[point_indices]

            # --- 2. PDE 物理配点采样 ---
            # 如果启用了 PDE 模式，独立抽取一组物理验证点 (放开 split 限制，允许验证集采样)
            if self.mode == "pde" and self.num_pde_points > 0:
                pde_indices = self._sample_pde_points()
                batch["coords_pde"] = coords_3d_full[pde_indices]
                # 提取这批 PDE 点对应的总场基线 (专家重点强调：约束总场必需)
                batch["A0_pde"] = torch.from_numpy(A0_target_full)[pde_indices]

        return batch