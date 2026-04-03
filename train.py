import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
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


class PhysicalWeightedSampler(Sampler):
    def __init__(self, weights, num_samples, excite_per_sample):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.excite_per_sample = excite_per_sample

    def __iter__(self):
        drawn_samples = torch.multinomial(self.weights, self.num_samples, replacement=True)
        for sample_idx in drawn_samples:
            for i in range(self.excite_per_sample):
                yield int(sample_idx * self.excite_per_sample + i)

    def __len__(self):
        return self.num_samples * self.excite_per_sample


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
    for p in model.parameters():
        p.requires_grad = False
    for p in model.heatmap_trunk.heat_logits_head.parameters():
        p.requires_grad = True


def train_epoch(model, dataloader, optimizer, scaler, heat_loss_fn, mse_loss_fn, mode, device, amp_enabled, epoch,
                epochs_stage1, is_stage3=False):
    model.train()
    total_heat_loss, total_A_loss = 0.0, 0.0

    dataloader.dataset.set_epoch(epoch)

    for batch in dataloader:
        v_pair = batch["v_pair"].to(device)
        coords_2d = batch["coords_2d"].to(device)
        heat_target = batch["heat_target"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=amp_enabled):
            branch_out = model.encode_observation(v_pair)
            heat_out = model.query_heatmap(branch_out, coords_2d)
            loss_heat = heat_loss_fn(heat_out["heat_logits"], heat_out["heat_prob"], heat_target)

            if mode == "heat_only" or is_stage3:
                loss = loss_heat
            elif mode == "joint":
                coords_3d = batch["coords_3d"].to(device)
                excite_idx = batch["excite_idx"].to(device)
                A_target = batch["A_target"].to(device)

                field_out = model.query_field(branch_out, coords_3d, excite_idx)
                loss_A = mse_loss_fn(field_out["A_delta_norm"], A_target)

                weight_A = 0.5 if (epoch - epochs_stage1) <= 8 else 1.0
                loss = (1.0 / 16.0) * loss_heat + weight_A * loss_A
                total_A_loss += loss_A.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_heat_loss += loss_heat.item()

    num_batches = len(dataloader)
    return total_heat_loss / num_batches, total_A_loss / num_batches if mode == "joint" else 0.0


@torch.no_grad()
def validate(model, dataloader, heat_loss_fn, mse_loss_fn, mode, device, val_grid, amp_enabled, skip_A_val,
             is_stage3=False):
    model.eval()
    total_heat_loss, total_A_loss = 0.0, 0.0
    total_iou, total_rmse = 0.0, 0.0
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

                    label_norm = batch["label_xywl"][b]
                    target_mask = get_grid_target(label_norm, val_grid).to(device)

                    total_heat_loss += heat_loss_fn(pred_logits, pred_prob, target_mask).item()
                    total_iou += calculate_iou(pred_prob, target_mask)

            if mode == "joint" and not skip_A_val and not is_stage3:
                coords_3d = batch["coords_3d"].to(device)
                A_target = batch["A_target"].to(device)

                field_out = model.query_field(branch_out, coords_3d, excite_idx)
                mse = mse_loss_fn(field_out["A_delta_norm"], A_target)

                total_A_loss += mse.item()
                total_rmse += torch.sqrt(mse).item()

    num_batches = len(dataloader)
    va_heat = total_heat_loss / val_physical_count if val_physical_count > 0 else 0.0
    va_iou = total_iou / val_physical_count if val_physical_count > 0 else 0.0

    if mode == "joint" and not skip_A_val and not is_stage3:
        va_A = total_A_loss / num_batches
        va_rmse = total_rmse / num_batches
    else:
        va_A, va_rmse = 0.0, 0.0

    return va_heat, va_A, va_iou, va_rmse


def get_dataloaders(h5_path, train_ids, val_ids, mode, batch_size, p2_train, p3_train, p2_val, p3_val, excite_train,
                    excite_val, use_weighted_sampler=False):
    print(
        f"\n🔄 装载 {mode.upper()} 管道 | BS={batch_size} | P3={p3_train}/{p3_val} | Excite={excite_train}/{excite_val} | Weighted={use_weighted_sampler}")

    train_dataset = EMT_SCF_Dataset(h5_path, sample_indices=train_ids, split="train", mode=mode,
                                    num_p2_points=p2_train, num_p3_points=p3_train, excite_per_sample=excite_train)
    val_dataset = EMT_SCF_Dataset(h5_path, sample_indices=val_ids, split="val", mode=mode,
                                  num_p2_points=p2_val, num_p3_points=p3_val, excite_per_sample=excite_val)

    if use_weighted_sampler:
        train_weights = train_dataset.get_physical_weights()
        sampler = PhysicalWeightedSampler(train_weights, train_dataset.num_samples, train_dataset.excite_per_sample)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                  persistent_workers=False, worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  persistent_workers=False, worker_init_fn=worker_init_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=False, worker_init_fn=worker_init_fn)
    return train_loader, val_loader


def main():
    print("🚀 启动 SCF-DeepONet 究极性能版 (物理级重构 V1.5 + 专家级课表)...")
    seed_everything(42)

    ckpt_dir, splits_dir = "checkpoints", "splits"
    os.makedirs(ckpt_dir, exist_ok=True);
    os.makedirs(splits_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda")

    h5_path = 'EMT_SCF_Dataset.h5'
    with h5py.File(h5_path, 'r') as hf:
        total_physical_samples = hf['v_pair'].shape[0]

    all_ids = np.arange(total_physical_samples)
    np.random.shuffle(all_ids)
    train_ids = all_ids[:850];
    val_ids = all_ids[850:]
    np.save(os.path.join(splits_dir, "train_ids.npy"), train_ids)
    np.save(os.path.join(splits_dir, "val_ids.npy"), val_ids)

    # 🚀 黄金课表：100 + 80 + 5
    epochs_stage1 = 100
    epochs_stage2 = 80
    epochs_stage3 = 5
    total_epochs = epochs_stage1 + epochs_stage2 + epochs_stage3

    bs_stage1 = 24
    p2_train_s1, p3_train_s1 = 2000, 100
    p2_val_s1, p3_val_s1 = 256, 100
    excite_train_s1, excite_val_s1 = 1, 1

    bs_stage2 = 8
    p2_train_s2, p3_train_s2 = 2000, 6000
    p2_val_s2, p3_val_s2 = 256, 2000
    excite_train_s2, excite_val_s2 = 4, 16

    # Stage 1 强制关闭加权，还原纯净的地基分布
    train_loader, val_loader = get_dataloaders(
        h5_path, train_ids, val_ids, mode="heat_only", batch_size=bs_stage1,
        p2_train=p2_train_s1, p3_train=p3_train_s1, p2_val=p2_val_s1, p3_val=p3_val_s1,
        excite_train=excite_train_s1, excite_val=excite_val_s1,
        use_weighted_sampler=False
    )
    val_grid = create_val_grid(device, resolution=128)

    model = SCFDeepONet_V1().to(device)
    heat_loss_fn = HeatmapLoss(dice_weight=0.5)
    mse_loss_fn = nn.MSELoss()

    init_lr = 3e-4
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scaler = GradScaler('cuda', enabled=amp_enabled)

    best_val_loss = float('inf')
    best_iou_stage1 = 0.0
    best_iou_stage2 = 0.0
    best_rmse_stage2 = float('inf')
    best_iou_stage3 = 0.0

    last_va_A, last_va_rmse = 0.0, 0.0

    print("\n================ 开始三阶段课表训练 (Curriculum Training) ================")
    for epoch in range(1, total_epochs + 1):
        start_time = time.time()

        is_stage3 = epoch > (epochs_stage1 + epochs_stage2)
        mode = "heat_only" if epoch <= epochs_stage1 or is_stage3 else "joint"

        # ================== 阶段 2：联合训练 ==================
        if epoch == epochs_stage1 + 1:
            print("\n🌟 触发进化！进入 Stage 2: 2D & 3D 联合训练阶段 🌟")
            # 🚀 强制加载 Stage 1 定位最准的权重 (最佳 IoU)
            best_heat_ckpt = os.path.join(ckpt_dir, "best_stage1_iou.pth")
            if os.path.exists(best_heat_ckpt):
                model.load_state_dict(torch.load(best_heat_ckpt, map_location=device)['model_state_dict'])
                print("✅ 权重回滚成功 (已加载 Stage 1 最佳 IoU 权重)！")

            best_val_loss = float('inf')
            for param_group in optimizer.param_groups: param_group['lr'] = init_lr
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

            del train_loader, val_loader
            # Stage 2 开启困难样本温和加权
            train_loader, val_loader = get_dataloaders(
                h5_path, train_ids, val_ids, mode="joint", batch_size=bs_stage2,
                p2_train=p2_train_s2, p3_train=p3_train_s2, p2_val=p2_val_s2, p3_val=p3_val_s2,
                excite_train=excite_train_s2, excite_val=excite_val_s2,
                use_weighted_sampler=True
            )

        # ================== 阶段 3：真正的热图精修 ==================
        if epoch == epochs_stage1 + epochs_stage2 + 1:
            print("\n🔥 终极冲刺！进入 Stage 3: Heatmap Polish (无 Scheduler 冻结版) 🔥")
            best_iou_ckpt = os.path.join(ckpt_dir, "best_stage2_iou.pth")
            if os.path.exists(best_iou_ckpt):
                model.load_state_dict(torch.load(best_iou_ckpt, map_location=device)['model_state_dict'])
                print(f"✅ 成功加载 Stage 2 最佳 IoU 权重进行精修！")

            freeze_all_for_stage3(model)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=3e-5, weight_decay=1e-5)

            del train_loader, val_loader
            # Stage 3 恢复纯自然分布
            train_loader, val_loader = get_dataloaders(
                h5_path, train_ids, val_ids, mode="heat_only", batch_size=bs_stage1,
                p2_train=p2_train_s1, p3_train=p3_train_s1, p2_val=p2_val_s1, p3_val=p3_val_s1,
                excite_train=excite_train_s1, excite_val=excite_val_s1,
                use_weighted_sampler=False
            )

        tr_heat, tr_A = train_epoch(model, train_loader, optimizer, scaler, heat_loss_fn, mse_loss_fn, mode, device,
                                    amp_enabled, epoch, epochs_stage1, is_stage3)

        skip_A_val = (mode == "joint") and (epoch % 2 != 0) and not is_stage3
        va_heat, va_A_curr, va_iou, va_rmse_curr = validate(model, val_loader, heat_loss_fn, mse_loss_fn, mode, device,
                                                            val_grid, amp_enabled, skip_A_val, is_stage3)

        if not skip_A_val and not is_stage3:
            last_va_A, last_va_rmse = va_A_curr, va_rmse_curr

        val_total_loss = va_heat if (mode == "heat_only" or is_stage3) else (1.0 / 16.0) * va_heat + last_va_A

        if not skip_A_val and not is_stage3:
            scheduler.step(val_total_loss)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        stage_name = "STAGE 3" if is_stage3 else mode.upper()
        skip_mark = " (Skipped)" if skip_A_val else ""
        print(
            f"Epoch [{epoch:03d}/{total_epochs:03d}] | Mode: {stage_name:9s} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        print(f"  [Train] Heat Loss: {tr_heat:.4f} | A_MSE Loss: {tr_A:.4e}")
        print(f"  [Val]   Heat Loss: {va_heat:.4f} | Heat IoU:   {va_iou * 100:05.2f}%")
        if mode == "joint" and not is_stage3:
            print(f"  [Val]   A_MSE:     {last_va_A:.4e}{skip_mark} | A_RMSE:     {last_va_rmse:.4e}{skip_mark}")

        checkpoint = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_total_loss, 'mode': mode
        }
        torch.save(checkpoint, os.path.join(ckpt_dir, "latest_model.pth"))

        if not skip_A_val:
            if is_stage3:
                if va_iou > best_iou_stage3:
                    best_iou_stage3 = va_iou
                    print(f"  🏆 Stage 3 热图登顶！保存为 best_stage3_iou.pth (IoU: {va_iou * 100:.2f}%)")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage3_iou.pth"))
            elif mode == "joint":
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    print(f"  🚀 综合最佳！保存为 best_stage2_joint.pth (Total Loss: {best_val_loss:.4e})")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage2_joint.pth"))
                if va_iou > best_iou_stage2:
                    best_iou_stage2 = va_iou
                    print(f"  🔥 热图最佳！保存为 best_stage2_iou.pth (IoU: {va_iou * 100:.2f}%)")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage2_iou.pth"))
                if last_va_rmse < best_rmse_stage2:
                    best_rmse_stage2 = last_va_rmse
                    print(f"  🌊 3D 场最佳！保存为 best_stage2_rmse.pth (RMSE: {last_va_rmse:.4e})")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage2_rmse.pth"))
            elif mode == "heat_only":
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    print(f"  🚀 Stage 1 Loss新纪录！保存为 best_stage1_heat.pth (Loss: {best_val_loss:.4e})")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage1_heat.pth"))
                if va_iou > best_iou_stage1:
                    best_iou_stage1 = va_iou
                    print(f"  🔥 Stage 1 热图最佳！保存为 best_stage1_iou.pth (IoU: {va_iou * 100:.2f}%)")
                    torch.save(checkpoint, os.path.join(ckpt_dir, "best_stage1_iou.pth"))


if __name__ == '__main__':
    main()