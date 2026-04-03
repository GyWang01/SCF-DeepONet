import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 基础组件 =================
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, num_freqs):
        super().__init__()
        self.in_dim = in_dim
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs) * torch.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        proj = x.unsqueeze(-1) * self.freqs
        proj = proj.reshape(*x.shape[:-1], -1)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class FiLMLayer(nn.Module):
    def __init__(self, latent_dim, feat_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, feat_dim * 2)
        nn.init.zeros_(self.fc.weight);
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, z):
        gamma, beta = self.fc(z).unsqueeze(1).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class DualFiLMLayer(nn.Module):
    def __init__(self, z_global_dim, z_exc_dim, feat_dim):
        super().__init__()
        self.fc_g = nn.Linear(z_global_dim, feat_dim * 2)
        self.fc_e = nn.Linear(z_exc_dim, feat_dim * 2)
        nn.init.zeros_(self.fc_g.weight);
        nn.init.zeros_(self.fc_g.bias)
        nn.init.zeros_(self.fc_e.weight);
        nn.init.zeros_(self.fc_e.bias)

    def forward(self, x, z_g, z_e):
        g1, b1 = self.fc_g(z_g).unsqueeze(1).chunk(2, dim=-1)
        g2, b2 = self.fc_e(z_e).unsqueeze(1).chunk(2, dim=-1)
        return x * (1.0 + g1 + g2) + (b1 + b2)


# ================= 分支网络 (Branch) =================
class PairEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, v_pair): return self.net(v_pair)


class ExcitationPool(nn.Module):
    def __init__(self, feat_dim=64, out_dim=128):
        super().__init__()
        self.score = nn.Linear(feat_dim, 1)
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, pair_feat):
        pair_proj = self.proj(pair_feat)
        attn = torch.sigmoid(self.score(pair_feat))
        return (pair_proj * attn).sum(dim=1) / (attn.sum(dim=1) + 1e-8)


class ExcitationMapRefiner(nn.Module):
    def __init__(self, c_exc=128, c_map=128, c_global=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_exc, c_map, 3, padding=1), nn.GroupNorm(8, c_map), nn.GELU(),
            nn.Conv2d(c_map, c_map, 3, padding=1), nn.GroupNorm(8, c_map), nn.GELU()
        )
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c_map, c_global), nn.LayerNorm(c_global)
        )

    def forward(self, z_e_all):
        B, E, C = z_e_all.shape
        x = z_e_all.view(B, 4, 4, C).permute(0, 3, 1, 2)
        F_map = self.conv(x)
        return F_map, self.global_head(F_map)


# ================= 主干网络 (Trunks) =================
class HeatmapTrunk(nn.Module):
    def __init__(self, c_map=128, c_global=256, c_heat=64, hidden=128, num_freqs=6):
        super().__init__()
        self.ff = FourierFeatures(2, num_freqs)
        in_dim = 2 + 2 * 2 * num_freqs + c_map

        self.in_layer = nn.Linear(in_dim, hidden)
        self.film1 = FiLMLayer(c_global, hidden)
        self.mid1 = nn.Linear(hidden, hidden)
        self.film2 = FiLMLayer(c_global, hidden)
        self.act = nn.GELU()

        self.heat_feat_head = nn.Linear(hidden, c_heat)
        # 【修改点 1】去掉 nn.Sigmoid()，只输出 logits
        self.heat_logits_head = nn.Linear(c_heat, 1)

    def forward(self, coords_2d_norm, f_xy, z_g):
        x = torch.cat([self.ff(coords_2d_norm), f_xy], dim=-1)
        x = self.act(self.film1(self.in_layer(x), z_g))
        x = self.act(self.film2(self.mid1(x), z_g))

        heat_feat = self.heat_feat_head(x)
        # 【修改点 2】分别计算 logits 和 prob
        heat_logits = self.heat_logits_head(heat_feat)
        heat_prob = torch.sigmoid(heat_logits)

        return heat_feat, heat_prob, heat_logits


class FieldTrunk(nn.Module):
    def __init__(self, c_map=128, c_heat=64, z_global=256, z_exc=128, hidden=256, num_freqs=8):
        super().__init__()
        self.ff = FourierFeatures(3, num_freqs)
        in_dim = 3 + 2 * 3 * num_freqs + c_map + c_heat + 1

        self.in_layer = nn.Linear(in_dim, hidden)
        self.film1 = DualFiLMLayer(z_global, z_exc, hidden)
        self.mid1 = nn.Linear(hidden, hidden)
        self.film2 = DualFiLMLayer(z_global, z_exc, hidden)
        self.mid2 = nn.Linear(hidden, hidden)
        self.film3 = DualFiLMLayer(z_global, z_exc, hidden)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden, 6)

    def forward(self, coords_3d_norm, f_xy, heat_feat, heat_prob, z_g, z_e):
        x = torch.cat([self.ff(coords_3d_norm), f_xy, heat_feat, heat_prob], dim=-1)
        x = self.act(self.film1(self.in_layer(x), z_g, z_e))
        x = self.act(self.film2(self.mid1(x), z_g, z_e))
        x = self.act(self.film3(self.mid2(x), z_g, z_e))
        return self.out(x)


# ================= 终极 SCF-DeepONet 主控类 =================
class SCFDeepONet_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair_encoder = PairEncoder()
        self.pool = ExcitationPool()
        self.refiner = ExcitationMapRefiner()
        self.heatmap_trunk = HeatmapTrunk()
        self.field_trunk = FieldTrunk()

        # 【终极修复点 5】物理标度硬编码为 Buffer，与网络权重永久绑定，绝不错位！
        self.register_buffer("max_field_xyz", torch.tensor([130.0, 90.0, 90.0], dtype=torch.float32))
        self.register_buffer("max_surf_xy", torch.tensor(32.0, dtype=torch.float32))

    def encode_observation(self, v_pair):
        B, num_exc, num_rec, feat_dim = v_pair.shape
        # 【终极修复点 4】杜绝 view 的内存不连续地雷，改为更安全的 reshape
        pair_feat = self.pair_encoder(v_pair.reshape(B * num_exc, num_rec, feat_dim))

        z_e_all = self.pool(pair_feat)
        z_e_all = z_e_all.view(B, num_exc, -1)

        F_map, z_g = self.refiner(z_e_all)
        return {"z_g": z_g, "z_e_all": z_e_all, "F_map": F_map}

    def sample_Fmap(self, F_map, coords_2d_norm):
        grid = coords_2d_norm.unsqueeze(2)
        feat = F.grid_sample(F_map, grid, align_corners=True, padding_mode='border')
        return feat.squeeze(-1).permute(0, 2, 1)

    def query_heatmap(self, branch_out, coords_2d_norm):
        f_xy = self.sample_Fmap(branch_out["F_map"], coords_2d_norm)
        # 接收三个返回值
        heat_feat, heat_prob, heat_logits = self.heatmap_trunk(coords_2d_norm, f_xy, branch_out["z_g"])
        # 将 logits 加入字典
        return {"heat_feat": heat_feat, "heat_prob": heat_prob, "heat_logits": heat_logits}

    def query_field(self, branch_out, coords_3d_field_norm, excite_idx):
        B = coords_3d_field_norm.shape[0]
        idx_b = torch.arange(B, device=excite_idx.device)
        z_e = branch_out["z_e_all"][idx_b, excite_idx]

        # 利用 Buffer 里的标度还原真实坐标并重投影到热图表面
        x_real = coords_3d_field_norm[..., 0] * self.max_field_xyz[0]
        y_real = coords_3d_field_norm[..., 1] * self.max_field_xyz[1]

        coords_2d_proj_norm = torch.stack([x_real / self.max_surf_xy, y_real / self.max_surf_xy], dim=-1)
        coords_2d_proj_norm = torch.clamp(coords_2d_proj_norm, -1.0, 1.0)

        heat_out = self.query_heatmap(branch_out, coords_2d_proj_norm)

        A_delta_norm = self.field_trunk(
            coords_3d_field_norm,
            self.sample_Fmap(branch_out["F_map"], coords_2d_proj_norm),
            heat_out["heat_feat"],
            heat_out["heat_prob"],
            branch_out["z_g"],
            z_e
        )
        return {
            "heat_feat_xy": heat_out["heat_feat"],
            "heat_prob_xy": heat_out["heat_prob"],
            "A_delta_norm": A_delta_norm
        }