import os
import torch
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from torch.amp import autocast

from dataset import EMT_SCF_Dataset
from networks import SCFDeepONet_V1
from train import create_val_grid, get_grid_target, calculate_iou


def get_pred_box(pred_prob, max_xy=32.0, res=128):
    prob_2d = pred_prob.view(res, res)
    mask = (prob_2d > 0.5)

    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    y_idx, x_idx = torch.where(mask)
    x_min, x_max = x_idx.min().item(), x_idx.max().item()
    y_min, y_max = y_idx.min().item(), y_idx.max().item()

    pix_size = 2.0 / (res - 1)
    x_min_norm = x_min * pix_size - 1.0 - pix_size / 2.0
    x_max_norm = x_max * pix_size - 1.0 + pix_size / 2.0
    y_min_norm = y_min * pix_size - 1.0 - pix_size / 2.0
    y_max_norm = y_max * pix_size - 1.0 + pix_size / 2.0

    cx = (x_min_norm + x_max_norm) / 2.0 * max_xy
    cy = (y_min_norm + y_max_norm) / 2.0 * max_xy
    w = (x_max_norm - x_min_norm) * max_xy
    l = (y_max_norm - y_min_norm) * max_xy
    return cx, cy, w, l


def evaluate_dataset_all(model, dataset_name, h5_path, sample_indices, device):
    print(f"\n================ 正在为 {dataset_name} 生成全景视觉报告 ================")

    # 建立输出文件夹
    out_dir = f"All_Test_Results"
    os.makedirs(out_dir, exist_ok=True)

    dataset = EMT_SCF_Dataset(h5_path, sample_indices=sample_indices, split="val",
                              mode="joint", num_p2_points=0, num_p3_points=8000, excite_per_sample=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    val_grid = create_val_grid(device, resolution=128)
    MAX_SURF_XY = 32.0

    sample_stats = {}
    global_item_idx = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            v_pair = batch["v_pair"].to(device)
            excite_idx = batch["excite_idx"].to(device)
            coords_3d = batch["coords_3d"].to(device)
            A_target = batch["A_target"].to(device)

            with autocast('cuda'):
                branch_out = model.encode_observation(v_pair)
                field_out = model.query_field(branch_out, coords_3d, excite_idx)

                B = v_pair.shape[0]
                for b in range(B):
                    physical_id = sample_indices[global_item_idx // 16]
                    excite_id = excite_idx[b].item()

                    if physical_id not in sample_stats:
                        sample_stats[physical_id] = {'field_mse_sum': 0.0, 'excite_count': 0}

                    mse = torch.nn.functional.mse_loss(field_out["A_delta_norm"][b], A_target[b]).item()
                    sample_stats[physical_id]['field_mse_sum'] += mse
                    sample_stats[physical_id]['excite_count'] += 1

                    if excite_id == 0:
                        single_branch = {k: v[b:b + 1] for k, v in branch_out.items()}
                        grid_input = val_grid.unsqueeze(0)
                        heat_out = model.query_heatmap(single_branch, grid_input)
                        pred_prob = heat_out["heat_prob"][0]

                        label_norm = batch["label_xywl"][b]
                        target_mask = get_grid_target(label_norm, val_grid).to(device)
                        iou = calculate_iou(pred_prob, target_mask)

                        sample_stats[physical_id].update({
                            'sample_id': physical_id,
                            'heat_iou': iou,
                            'plot_data': {
                                'grid': val_grid.cpu().numpy(),
                                'pred_prob': pred_prob.cpu().numpy(),
                                'target_mask': target_mask.cpu().numpy(),
                                'coords_3d': coords_3d[b].cpu().numpy(),
                                'pred_A': field_out["A_delta_norm"][b].cpu().numpy(),
                                'target_A': A_target[b].cpu().numpy(),
                                'label': label_norm.cpu().numpy()
                            }
                        })
                    global_item_idx += 1

    records = []
    for pid, stats in sample_stats.items():
        records.append(stats)

    df = pd.DataFrame(records)

    # 🚀 按 IoU 降序排列，方便逐帧查看
    df_sorted = df.sort_values('heat_iou', ascending=False).reset_index(drop=True)

    print(f"📸 准备生成 {len(df_sorted)} 张评估图片...")
    for rank, row in df_sorted.iterrows():
        # 命名格式：Rank_01_Sample_42_IoU_99.5%
        filename = f"Rank_{rank + 1:02d}_Sample_{int(row['sample_id']):03d}_(IoU_{row['heat_iou'] * 100:.1f}%)"
        filepath = os.path.join(out_dir, filename)

        plot_visualizations(row['plot_data'], filepath, filename)

    print(f"✅ 全景评估完成！所有图片已保存在 '{out_dir}/' 目录下。")


def plot_visualizations(data, filepath, title):
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    ax1 = fig.add_subplot(1, 3, 1)
    grid = data['grid'].reshape(128, 128, 2)
    X, Y = grid[..., 0], grid[..., 1]
    Pred = data['pred_prob'].reshape(128, 128)

    im1 = ax1.pcolormesh(X, Y, Pred, shading='auto', cmap='magma', vmin=0, vmax=1)
    cx, cy, w, l = data['label']
    rect = plt.Rectangle((cx - w / 2, cy - l / 2), w, l, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax1.add_patch(rect)
    ax1.set_title("Predicted Heatmap & True Box (Red)")
    plt.colorbar(im1, ax=ax1)

    true_A = data['target_A'][:, 0]
    pred_A = data['pred_A'][:, 0]
    vmin, vmax = min(true_A.min(), pred_A.min()), max(true_A.max(), pred_A.max())

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    sc2 = ax2.scatter(data['coords_3d'][:, 0], data['coords_3d'][:, 1], data['coords_3d'][:, 2],
                      c=true_A, cmap='viridis', s=2, vmin=vmin, vmax=vmax)
    ax2.set_title(f"Ground Truth A-Field (Comp 0)\nRange: [{vmin:.2e}, {vmax:.2e}]")

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    sc3 = ax3.scatter(data['coords_3d'][:, 0], data['coords_3d'][:, 1], data['coords_3d'][:, 2],
                      c=pred_A, cmap='viridis', s=2, vmin=vmin, vmax=vmax)
    ax3.set_title(f"Predicted A-Field (Comp 0)\nRange: [{vmin:.2e}, {vmax:.2e}]")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    fig.colorbar(sc3, cax=cbar_ax)

    plt.savefig(f"{filepath}.jpg", dpi=200)  # dpi设为200足够清晰且生成快
    plt.close()


def main():
    print("🔬 启动 SCF-DeepONet 全景画图脚本...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SCFDeepONet_V1().to(device)

    # 🚀 强制加载我们 V1.5 辛辛苦苦炼出来的最强权重
    ckpt_path = "checkpoints/best_stage3_iou.pth"
    print(f"📦 正在加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 成功加载保存的模型！")

    h5_path_test = 'EMT_SCF_Test_Dataset.h5'
    if os.path.exists(h5_path_test):
        with h5py.File(h5_path_test, 'r') as hf:
            total_test_samples = hf['v_pair'].shape[0]
        test_ids = np.arange(total_test_samples)
        evaluate_dataset_all(model, "Secret_Test_Set", h5_path_test, test_ids, device)
    else:
        print("\n⚠️ 找不到 EMT_SCF_Test_Dataset.h5！")


if __name__ == '__main__':
    main()