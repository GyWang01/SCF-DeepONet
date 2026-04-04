import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import time
import h5py
import random

from dataset import EMT_SCF_Dataset
from networks import SCFDeepONet_V1


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HeatmapLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, probs, target):
        bce_loss = self.bce_logits(logits, target)
        smooth = 1e-6
        intersection = (probs * target).sum()
        dice_loss = 1.0 - (2. * intersection + smooth) / (probs.sum() + target.sum() + smooth)
        return bce_loss + self.dice_weight * dice_loss


def calculate_iou(pred_prob, target_mask, threshold=0.5):
    pred_bin = (pred_prob > threshold).float()
    target_bin = (target_mask > 0.5).float()
    intersection = (pred_bin * target_bin).sum().item()
    union = (pred_bin + target_bin).sum().item() - intersection
    if union == 0: return 1.0
    return intersection / union


def create_val_grid(device, resolution=128):
    lin = torch.linspace(-1.0, 1.0, resolution, device=device)
    Y, X = torch.meshgrid(lin, lin, indexing='ij')
    grid_2d = torch.stack([X, Y], dim=-1).view(-1, 2)
    return grid_2d


def get_grid_target(label_norm, grid_2d):
    cx, cy, w, l = label_norm.tolist()
    x_min, x_max = cx - w / 2, cx + w / 2
    y_min, y_max = cy - l / 2, cy + l / 2
    in_x = (grid_2d[:, 0] >= x_min) & (grid_2d[:, 0] <= x_max)
    in_y = (grid_2d[:, 1] >= y_min) & (grid_2d[:, 1] <= y_max)
    return (in_x & in_y).float().unsqueeze(1)


def freeze_all_for_stage3(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.heatmap_trunk.heat_logits_head.parameters(): p.requires_grad = True


def freeze_for_stage4_pde(model):
    for p in model.parameters(): p.requires_grad = False
    for p in model.field_trunk.parameters(): p.requires_grad = True


def train_epoch(model, dataloader, optimizer, scaler, heat_loss_fn, mse_loss_fn, mode, device, amp_enabled, epoch,
                epochs_stage1, A_scale, start_stage4_epoch, is_stage3=False):
    model.train()
    total_heat_loss, total_A_loss, total_pde_loss = 0.0, 0.0, 0.0
    epoch_raw_rms, epoch_rel_rms = 0.0, 0.0

    dataloader.dataset.set_epoch(epoch)

    for batch in dataloader:
        v_pair = batch["v_pair"].to(device)
        coords_2d = batch["coords_2d"].to(device)
        heat_target = batch["heat_target"].to(device)
        label_norm = batch["label_xywl"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=amp_enabled):
            branch_out = model.encode_observation(v_pair)
            heat_out = model.query_heatmap(branch_out, coords_2d)
            loss_heat = heat_loss_fn(heat_out["heat_logits"], heat_out["heat_prob"], heat_target)

            if mode in ["joint", "pde"]:
                coords_3d = batch["coords_3d"].to(device)
                excite_idx = batch["excite_idx"].to(device)
                A_target = batch["A_target"].to(device)

                field_out = model.query_field(branch_out, coords_3d, excite_idx)
                loss_A = mse_loss_fn(field_out["A_delta_norm"], A_target)
                weight_A = 0.5 if (epoch - epochs_stage1) <= 8 else 1.0

                if mode == "joint":
                    loss = (1.0 / 16.0) * loss_heat + weight_A * loss_A
            else:
                loss = loss_heat

        if mode == "pde":
            coords_pde = batch["coords_pde"].to(device)
            A0_pde = batch["A0_pde"].to(device)
            defect_h = batch["defect_h"].to(device)

            branch_out_fp32 = {k: v.float() for k, v in branch_out.items()}

            with autocast('cuda', enabled=False):
                loss_pde_val, loss_div_val, raw_rms, rel_rms = model.compute_pde_loss(
                    branch_out_fp32, coords_pde.float(), A0_pde.float(),
                    excite_idx, label_norm.float(), defect_h.float(), A_scale.float()
                )

            # 🚀 专家方案：加入 Warmup 机制！前 5 个 Epoch 线性增长到 1e-4
            current_pde_epoch = epoch - start_stage4_epoch + 1
            warmup_factor = min(current_pde_epoch / 5.0, 1.0)

            lambda_pde = 1e-4 * warmup_factor
            lambda_div = 0.0

            loss = (1.0 / 16.0) * loss_heat + weight_A * loss_A + lambda_pde * loss_pde_val + lambda_div * loss_div_val

            total_pde_loss += loss_pde_val.item()
            epoch_raw_rms += raw_rms
            epoch_rel_rms += rel_rms

        if mode in ["joint", "pde"]:
            total_A_loss += loss_A.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_heat_loss += loss_heat.item()

    num_batches = len(dataloader)
    avg_pde = total_pde_loss / num_batches if mode == "pde" else 0.0
    avg_A = total_A_loss / num_batches if mode in ["joint", "pde"] else 0.0
    diags = (epoch_raw_rms / num_batches, epoch_rel_rms / num_batches) if mode == "pde" else None

    return total_heat_loss / num_batches, avg_A, avg_pde, diags


@torch.no_grad()  # 默认全局无梯度
def validate(model, dataloader, heat_loss_fn, mse_loss_fn, mode, device, val_grid, amp_enabled, skip_A_val, A_scale,
             is_stage3=False):
    model.eval()
    total_heat_loss, total_A_loss, total_iou, total_rmse = 0.0, 0.0, 0.0, 0.0
    total_val_raw_rms, total_val_rel_rms = 0.0, 0.0
    val_physical_count = 0

    for batch in dataloader:
        v_pair = batch["v_pair"].to(device)
        excite_idx = batch["excite_idx"].to(device)

        with autocast('cuda', enabled=amp_enabled):
            branch_out = model.encode_observation(v_pair)

            B = v_pair.shape[0]
            for b in range(B):
                if excite_idx[b].item() == 0:
                    val_physical_count += 1
                    single_branch = {k: v[b:b + 1] for k, v in branch_out.items()}
                    grid_input = val_grid.unsqueeze(0)

                    heat_out = model.query_heatmap(single_branch, grid_input)
                    pred_logits = heat_out["heat_logits"][0]
                    pred_prob = heat_out["heat_prob"][0]
                    target_mask = get_grid_target(batch["label_xywl"][b], val_grid).to(device)

                    total_heat_loss += heat_loss_fn(pred_logits, pred_prob, target_mask).item()
                    total_iou += calculate_iou(pred_prob, target_mask)

            if mode in ["joint", "pde"] and not skip_A_val and not is_stage3:
                coords_3d = batch["coords_3d"].to(device)
                A_target = batch["A_target"].to(device)

                field_out = model.query_field(branch_out, coords_3d, excite_idx)
                mse = mse_loss_fn(field_out["A_delta_norm"], A_target)
                total_A_loss += mse.item()
                total_rmse += torch.sqrt(mse).item()

        # 🚀 专家方案：在验证集独立计算 PDE 残差
        if mode == "pde" and not skip_A_val and not is_stage3:
            coords_pde = batch["coords_pde"].to(device)
            A0_pde = batch["A0_pde"].to(device)
            defect_h = batch["defect_h"].to(device)
            label_norm = batch["label_xywl"].to(device)

            branch_out_fp32 = {k: v.float() for k, v in branch_out.items()}

            # 🧨 技术难点破解：局部强行开启 Autograd，否则无法计算二阶偏导
            with torch.enable_grad():
                with autocast('cuda', enabled=False):
                    _, _, raw_rms, rel_rms = model.compute_pde_loss(
                        branch_out_fp32, coords_pde.float(), A0_pde.float(),
                        excite_idx, label_norm.float(), defect_h.float(), A_scale.float()
                    )
            total_val_raw_rms += raw_rms
            total_val_rel_rms += rel_rms

    num_batches = len(dataloader)
    va_heat = total_heat_loss / val_physical_count if val_physical_count > 0 else 0.0
    va_iou = total_iou / val_physical_count if val_physical_count > 0 else 0.0

    if mode in ["joint", "pde"] and not skip_A_val and not is_stage3:
        va_A = total_A_loss / num_batches
        va_rmse = total_rmse / num_batches
    else:
        va_A, va_rmse = 0.0, 0.0

    val_diags = (total_val_raw_rms / num_batches, total_val_rel_rms / num_batches) if mode == "pde" else None

    return va_heat, va_A, va_iou, va_rmse, val_diags


def get_dataloaders(h5_path, train_ids, val_ids, mode, batch_size, p2_train, p3_train, p2_val, p3_val, excite_train,
                    excite_val):
    num_pde = 256 if mode == "pde" else 0
    # 🚀 确保验证集也分配了同样的 PDE 计算点
    val_num_pde = 256 if mode == "pde" else 0

    n_workers = 0 if mode == "pde" else 4
    pers_workers = False if mode == "pde" else True

    print(
        f"\n🔄 装载 {mode.upper()} 管道 | BS={batch_size} | P3={p3_train}/{p3_val} | PDE_Pts={num_pde}/{val_num_pde} | Excite={excite_train}/{excite_val}")

    train_dataset = EMT_SCF_Dataset(h5_path, sample_indices=train_ids, split="train", mode=mode,
                                    num_p2_points=p2_train, num_p3_points=p3_train, num_pde_points=num_pde,
                                    excite_per_sample=excite_train)
    val_dataset = EMT_SCF_Dataset(h5_path, sample_indices=val_ids, split="val", mode=mode,
                                  num_p2_points=p2_val, num_p3_points=p3_val, num_pde_points=val_num_pde,
                                  excite_per_sample=excite_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
                              pin_memory=True, persistent_workers=pers_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    return train_loader, val_loader


def main():
    print("🚀 启动 SCF-DeepONet 究极性能版训练引擎 (PDE 专家终极打磨版)...")
    seed_everything(42)

    ckpt_dir, splits_dir = "checkpoints", "splits"
    os.makedirs(ckpt_dir, exist_ok=True);
    os.makedirs(splits_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda")

    h5_path = 'EMT_SCF_Dataset.h5'
    with h5py.File(h5_path, 'r') as hf:
        total_physical_samples = hf['v_pair'].shape[0]
        A_scale_val = torch.tensor(hf.attrs['A_scale'], dtype=torch.float32, device=device)

    all_ids = np.arange(total_physical_samples)
    np.random.shuffle(all_ids)
    train_ids = all_ids[:850];
    val_ids = all_ids[850:]

    epochs_stage1 = 20
    epochs_stage2 = 80
    epochs_stage3 = 5
    epochs_stage4 = 30  # 🚀 专家方案：拉长 PDE 磨合期到 30 个 Epoch
    total_epochs = epochs_stage1 + epochs_stage2 + epochs_stage3 + epochs_stage4

    bs_stage1, bs_stage2 = 24, 8
    p2_train_s1, p3_train_s1 = 2000, 100
    p2_val_s1, p3_val_s1 = 256, 100
    excite_train_s1, excite_val_s1 = 16, 16

    p2_train_s2, p3_train_s2 = 2000, 6000
    p2_val_s2, p3_val_s2 = 256, 2000
    excite_train_s2, excite_val_s2 = 4, 16

    train_loader, val_loader = get_dataloaders(
        h5_path, train_ids, val_ids, mode="heat_only", batch_size=bs_stage1,
        p2_train=p2_train_s1, p3_train=p3_train_s1, p2_val=p2_val_s1, p3_val=p3_val_s1,
        excite_train=excite_train_s1, excite_val=excite_val_s1
    )
    val_grid = create_val_grid(device, resolution=128)

    model = SCFDeepONet_V1().to(device)
    heat_loss_fn = HeatmapLoss(dice_weight=0.5)
    mse_loss_fn = nn.MSELoss()

    init_lr = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
    scaler = GradScaler('cuda', enabled=amp_enabled)

    best_val_loss, best_iou_stage2, best_rmse_stage2 = float('inf'), 0.0, float('inf')
    best_val_loss_stage3, best_rmse_stage4, best_joint_stage4 = float('inf'), float('inf'), float('inf')

    last_va_A, last_va_rmse = 0.0, 0.0

    print("\n================ 开始四阶段课表训练 (Curriculum Training w/ PDE) ================")
    START_EPOCH = 106
    start_stage4_epoch = epochs_stage1 + epochs_stage2 + epochs_stage3 + 1

    for epoch in range(START_EPOCH, total_epochs + 1):
        start_time = time.time()

        is_stage3 = (epoch > epochs_stage1 + epochs_stage2) and (epoch <= epochs_stage1 + epochs_stage2 + epochs_stage3)
        is_stage4 = epoch > (epochs_stage1 + epochs_stage2 + epochs_stage3)

        if epoch <= epochs_stage1 or is_stage3:
            mode = "heat_only"
        elif is_stage4:
            mode = "pde"
        else:
            mode = "joint"

        if epoch == epochs_stage1 + 1:
            print("\n🌟 进入 Stage 2: 2D & 3D 联合训练阶段 🌟")
            best_heat_ckpt = os.path.join(ckpt_dir, "best_stage1_heat.pth")
            if os.path.exists(best_heat_ckpt):
                model.load_state_dict(torch.load(best_heat_ckpt, map_location=device)['model_state_dict'], strict=False)
            for param_group in optimizer.param_groups: param_group['lr'] = init_lr
            del train_loader, val_loader
            train_loader, val_loader = get_dataloaders(h5_path, train_ids, val_ids, mode="joint", batch_size=bs_stage2,
                                                       p2_train=p2_train_s2, p3_train=p3_train_s2, p2_val=p2_val_s2,
                                                       p3_val=p3_val_s2, excite_train=excite_train_s2,
                                                       excite_val=excite_val_s2)

        if epoch == epochs_stage1 + epochs_stage2 + 1:
            print("\n🔥 进入 Stage 3: Heatmap Polish 绝对冻结版 🔥")
            best_iou_ckpt = os.path.join(ckpt_dir, "best_stage2_iou.pth")
            if os.path.exists(best_iou_ckpt):
                model.load_state_dict(torch.load(best_iou_ckpt, map_location=device)['model_state_dict'], strict=False)
            freeze_all_for_stage3(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=3e-5, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            del train_loader, val_loader
            train_loader, val_loader = get_dataloaders(h5_path, train_ids, val_ids, mode="heat_only",
                                                       batch_size=bs_stage1, p2_train=p2_train_s1, p3_train=p3_train_s1,
                                                       p2_val=p2_val_s1, p3_val=p3_val_s1, excite_train=excite_train_s1,
                                                       excite_val=excite_val_s1)

        if epoch == start_stage4_epoch:
            print("\n⚛️ 数据-物理双驱动！进入 Stage 4: 专家精调路线 (Warmup + 30 Epochs) ⚛️")
            best_rmse_ckpt = os.path.join(ckpt_dir, "best_stage2_rmse.pth")
            if os.path.exists(best_rmse_ckpt):
                model.load_state_dict(torch.load(best_rmse_ckpt, map_location=device)['model_state_dict'], strict=False)
                print(f"✅ 成功回滚至 Stage 2 最佳 RMSE 权重，作为 PDE 基底！")

            freeze_for_stage4_pde(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            optimizer = optim.AdamW(trainable_params, lr=1e-5, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

            del train_loader, val_loader
            train_loader, val_loader = get_dataloaders(h5_path, train_ids, val_ids, mode="pde", batch_size=bs_stage2,
                                                       p2_train=p2_train_s2, p3_train=p3_train_s2, p2_val=p2_val_s2,
                                                       p3_val=p3_val_s2, excite_train=excite_train_s2,
                                                       excite_val=excite_val_s2)

        tr_heat, tr_A, tr_pde, tr_diags = train_epoch(
            model, train_loader, optimizer, scaler, heat_loss_fn, mse_loss_fn,
            mode, device, amp_enabled, epoch, epochs_stage1, A_scale_val, start_stage4_epoch, is_stage3
        )

        skip_A_val = (mode in ["joint", "pde"]) and (epoch % 2 != 0) and not is_stage3
        va_heat, va_A_curr, va_iou, va_rmse_curr, va_diags = validate(
            model, val_loader, heat_loss_fn, mse_loss_fn, mode, device, val_grid, amp_enabled, skip_A_val, A_scale_val,
            is_stage3
        )

        if not skip_A_val and not is_stage3: last_va_A, last_va_rmse = va_A_curr, va_rmse_curr

        val_total_loss = va_heat if (mode == "heat_only" or is_stage3) else (1.0 / 16.0) * va_heat + last_va_A
        if not skip_A_val: scheduler.step(val_total_loss)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        stage_name = f"STAGE 4" if is_stage4 else f"STAGE 3" if is_stage3 else f"STAGE 2"
        skip_mark = " (Skipped)" if skip_A_val else ""

        # 🚀 专家点名的打印日志体系
        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] | {stage_name} ({mode.upper()}) | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # 打印当前计算出的 Warmup λ (仅供显示)
        curr_lambda = 1e-4 * min((epoch - start_stage4_epoch + 1) / 5.0, 1.0) if mode == "pde" else 0.0

        print(f"  [Train] Heat: {tr_heat:.4f} | A_MSE: {tr_A:.4e}" + (
            f" | PDE(Rel): {tr_pde:.4f} (λ={curr_lambda:.1e})" if mode == "pde" else ""))
        if mode == "pde" and tr_diags is not None:
            print(f"          > Train Diags: Raw RMS = {tr_diags[0]:.2e} | Rel RMS = {tr_diags[1]:.4f}")

        print(f"  [Val]   Heat: {va_heat:.4f} | IoU: {va_iou * 100:05.2f}%")
        if mode in ["joint", "pde"] and not is_stage3:
            print(f"  [Val]   A_MSE: {last_va_A:.4e}{skip_mark} | A_RMSE: {last_va_rmse:.4e}{skip_mark}")
            if mode == "pde" and va_diags is not None and not skip_A_val:
                print(f"          > Val Diags  : Raw RMS = {va_diags[0]:.2e} | Rel RMS = {va_diags[1]:.4f} 🎯")

        checkpoint = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_total_loss, 'mode': mode
        }
        torch.save(checkpoint, os.path.join(ckpt_dir, "latest_model.pth"))

        if not skip_A_val:
            if is_stage4:
                if last_va_rmse < best_rmse_stage4:
                    best_rmse_stage4 = last_va_rmse
                    print(f"  ⚛️ PDE 精调新纪录！保存为 best_stage4_pde_rmse.pth (RMSE: {last_va_rmse:.4e})")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage4_pde_rmse.pth"))
                if val_total_loss < best_joint_stage4:
                    best_joint_stage4 = val_total_loss
                    print(f"  ⚖️ PDE 综合最佳！保存为 best_stage4_pde_joint.pth")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage4_pde_joint.pth"))


if __name__ == '__main__':
    main()