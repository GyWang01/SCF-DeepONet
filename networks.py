import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 基础组件 (保持不变) =================
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
        self.heat_logits_head = nn.Linear(c_heat, 1)

    def forward(self, coords_2d_norm, f_xy, z_g):
        x = torch.cat([self.ff(coords_2d_norm), f_xy], dim=-1)
        x = self.act(self.film1(self.in_layer(x), z_g))
        x = self.act(self.film2(self.mid1(x), z_g))

        heat_feat = self.heat_feat_head(x)
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


class SCFDeepONet_V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pair_encoder = PairEncoder()
        self.pool = ExcitationPool()
        self.refiner = ExcitationMapRefiner()
        self.heatmap_trunk = HeatmapTrunk()
        self.field_trunk = FieldTrunk()

        self.register_buffer("max_field_xyz", torch.tensor([130.0, 90.0, 90.0], dtype=torch.float32))
        self.register_buffer("max_surf_xy", torch.tensor(32.0, dtype=torch.float32))

        # 🚀 物理单位修正：必须还原回真正的“米” (m) 来求导
        self.register_buffer("max_field_xyz_m", torch.tensor([130e-3, 90e-3, 90e-3], dtype=torch.float32))
        self.register_buffer("max_surf_xy_m", torch.tensor(32e-3, dtype=torch.float32))

        self.pde_params = {
            "omega": 2.0 * torch.pi * 100000.0,  # 🚀 修复点：100 kHz
            "mu_0": 4 * torch.pi * 1e-7,
            "mu_r_iron": 100.0,
            "sigma_iron": 1e7,
            "sigma_air": 1e-6
        }

    def encode_observation(self, v_pair):
        B, num_exc, num_rec, feat_dim = v_pair.shape
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
        heat_feat, heat_prob, heat_logits = self.heatmap_trunk(coords_2d_norm, f_xy, branch_out["z_g"])
        return {"heat_feat": heat_feat, "heat_prob": heat_prob, "heat_logits": heat_logits}

    def query_field(self, branch_out, coords_3d_field_norm, excite_idx):
        B = coords_3d_field_norm.shape[0]
        idx_b = torch.arange(B, device=excite_idx.device)
        z_e = branch_out["z_e_all"][idx_b, excite_idx]

        x_real = coords_3d_field_norm[..., 0] * self.max_field_xyz[0]
        y_real = coords_3d_field_norm[..., 1] * self.max_field_xyz[1]
        coords_2d_proj_norm = torch.stack([x_real / self.max_surf_xy, y_real / self.max_surf_xy], dim=-1)
        coords_2d_proj_norm = torch.clamp(coords_2d_proj_norm, -1.0, 1.0)

        # 🚀 截断梯度，防止 grid_sample 二阶导报错
        coords_2d_proj_norm = coords_2d_proj_norm.detach()

        heat_out = self.query_heatmap(branch_out, coords_2d_proj_norm)
        A_delta_norm = self.field_trunk(
            coords_3d_field_norm, self.sample_Fmap(branch_out["F_map"], coords_2d_proj_norm),
            heat_out["heat_feat"], heat_out["heat_prob"], branch_out["z_g"], z_e
        )
        return {"heat_feat_xy": heat_out["heat_feat"], "heat_prob_xy": heat_out["heat_prob"],
                "A_delta_norm": A_delta_norm}

    def _compute_soft_mask(self, coords_real_m, label_xywl, defect_h_mm, sharpness=5.0):
        defect_h_m = (defect_h_mm * 1e-3).view(-1, 1, 1)
        cx = label_xywl[:, 0:1] * self.max_surf_xy_m
        cy = label_xywl[:, 1:2] * self.max_surf_xy_m
        w = label_xywl[:, 2:3] * self.max_surf_xy_m
        l = label_xywl[:, 3:4] * self.max_surf_xy_m

        cx, cy, w, l = [t.unsqueeze(1) for t in (cx, cy, w, l)]
        x, y, z = coords_real_m[..., 0:1], coords_real_m[..., 1:2], coords_real_m[..., 2:3]

        x_min, x_max = cx - w / 2, cx + w / 2
        y_min, y_max = cy - l / 2, cy + l / 2
        z_max, z_min = 0.0, -defect_h_m

        sharp_m = sharpness * 1000.0
        mx = torch.sigmoid(sharp_m * (x - x_min)) * torch.sigmoid(sharp_m * (x_max - x))
        my = torch.sigmoid(sharp_m * (y - y_min)) * torch.sigmoid(sharp_m * (y_max - y))
        mz = torch.sigmoid(sharp_m * (z - z_min)) * torch.sigmoid(sharp_m * (z_max - z))
        return mx * my * mz

    def _curl(self, A_vec, x_norm):
        Ax, Ay, Az = A_vec[..., 0:1], A_vec[..., 1:2], A_vec[..., 2:3]
        dAx_dxnorm = torch.autograd.grad(Ax, x_norm, torch.ones_like(Ax), create_graph=True)[0]
        dAy_dxnorm = torch.autograd.grad(Ay, x_norm, torch.ones_like(Ay), create_graph=True)[0]
        dAz_dxnorm = torch.autograd.grad(Az, x_norm, torch.ones_like(Az), create_graph=True)[0]

        # 🚀 除以米制标度，回归国际单位系
        scale = self.max_field_xyz_m.view(1, 1, 3)
        dAx_dreal, dAy_dreal, dAz_dreal = dAx_dxnorm / scale, dAy_dxnorm / scale, dAz_dxnorm / scale

        curl_x = dAz_dreal[..., 1:2] - dAy_dreal[..., 2:3]
        curl_y = dAx_dreal[..., 2:3] - dAz_dreal[..., 0:1]
        curl_z = dAy_dreal[..., 0:1] - dAx_dreal[..., 1:2]
        return torch.cat([curl_x, curl_y, curl_z], dim=-1), (dAx_dreal, dAy_dreal, dAz_dreal)

    def compute_pde_loss(self, branch_out, coords_pde_norm, A0_pde_norm, excite_idx, label_xywl, defect_h, A_scale):
        coords_pde_norm.requires_grad_(True)

        field_out = self.query_field(branch_out, coords_pde_norm, excite_idx)
        A_total_norm = A0_pde_norm + field_out["A_delta_norm"]
        A_total_real = A_total_norm * A_scale

        Ar = A_total_real[..., [0, 2, 4]]
        Ai = A_total_real[..., [1, 3, 5]]

        coords_real_m = coords_pde_norm * self.max_field_xyz_m.view(1, 1, 3)
        mask_def = self._compute_soft_mask(coords_real_m, label_xywl, defect_h)

        mu = self.pde_params["mu_0"] * (self.pde_params["mu_r_iron"] * (1 - mask_def) + 1.0 * mask_def)
        sigma = self.pde_params["sigma_iron"] * (1 - mask_def) + self.pde_params["sigma_air"] * mask_def

        nu = 1.0 / mu
        omega = self.pde_params["omega"]

        curl_Ar, J_Ar = self._curl(Ar, coords_pde_norm)
        curl_Ai, J_Ai = self._curl(Ai, coords_pde_norm)

        curl_nu_curl_Ar, _ = self._curl(nu * curl_Ar, coords_pde_norm)
        curl_nu_curl_Ai, _ = self._curl(nu * curl_Ai, coords_pde_norm)

        # 🚀 A计划：保持原始 SI 残差（不引入相对归一化）
        R_r = curl_nu_curl_Ar - omega * sigma * Ai
        R_i = curl_nu_curl_Ai + omega * sigma * Ar
        loss_pde = torch.mean(R_r ** 2 + R_i ** 2)

        div_Ar = J_Ar[0][..., 0:1] + J_Ar[1][..., 1:2] + J_Ar[2][..., 2:3]
        div_Ai = J_Ai[0][..., 0:1] + J_Ai[1][..., 1:2] + J_Ai[2][..., 2:3]
        loss_div = torch.mean(div_Ar ** 2 + div_Ai ** 2)

        return loss_pde, loss_div