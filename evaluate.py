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


# =====================================================================
# 🚀 新增功能：从网络输出的概率热图中，严格反推出物理预测框
# =====================================================================
def get_pred_box(pred_prob, max_xy=32.0, res=128):
    # 【修复点】MLP 输出的是展平的 16384 个点，必须先揉回 128x128 的 2D 图像！
    prob_2d = pred_prob.view(res, res)
    mask = (prob_2d > 0.5)

    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    y_idx, x_idx = torch.where(mask)
    x_min, x_max = x_idx.min().item(), x_idx.max().item()
    y_min, y_max = y_idx.min().item(), y_idx.max().item()

    # 考虑像素本身的物理宽度，进行严谨的边界补偿
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


def evaluate_dataset(model, dataset_name, h5_path, sample_indices, device, plot_top_worst=True):
    print(f"\n================ 正在生成 {dataset_name} 量化报告 ================")

    dataset = EMT_SCF_Dataset(h5_path, sample_indices=sample_indices, split="val",
                              mode="joint", num_p2_points=0, num_p3_points=8000, excite_per_sample=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    val_grid = create_val_grid(device, resolution=128)
    MAX_SURF_XY = 32.0

    sample_stats = {}  # 用于收集 CSV 数据字典
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
                # 批量计算 3D 场
                field_out = model.query_field(branch_out, coords_3d, excite_idx)

                B = v_pair.shape[0]
                for b in range(B):
                    physical_id = sample_indices[global_item_idx // 16]
                    excite_id = excite_idx[b].item()

                    if physical_id not in sample_stats:
                        sample_stats[physical_id] = {'field_mse_sum': 0.0, 'excite_count': 0}

                    # 🚀 累计计算 3D A 场的均方误差 (对16个激励求平均)
                    mse = torch.nn.functional.mse_loss(field_out["A_delta_norm"][b], A_target[b]).item()
                    sample_stats[physical_id]['field_mse_sum'] += mse
                    sample_stats[physical_id]['excite_count'] += 1

                    # 🚀 仅当 excite_idx == 0 时，进行 2D 热图计算（同一缺陷热图共享）
                    if excite_id == 0:
                        single_branch = {k: v[b:b + 1] for k, v in branch_out.items()}
                        grid_input = val_grid.unsqueeze(0)
                        heat_out = model.query_heatmap(single_branch, grid_input)
                        pred_prob = heat_out["heat_prob"][0]

                        label_norm = batch["label_xywl"][b]
                        target_mask = get_grid_target(label_norm, val_grid).to(device)
                        iou = calculate_iou(pred_prob, target_mask)

                        # 解析真实标签物理值
                        true_cx, true_cy, true_w, true_l = (label_norm.cpu().numpy() * MAX_SURF_XY)
                        # 解析预测标签物理值
                        pred_cx, pred_cy, pred_w, pred_l = get_pred_box(pred_prob.cpu(), MAX_SURF_XY)

                        # 计算各种细粒度误差
                        if np.isnan(pred_cx):
                            c_err, w_err, l_err = np.nan, np.nan, np.nan
                        else:
                            c_err = np.sqrt((pred_cx - true_cx) ** 2 + (pred_cy - true_cy) ** 2)
                            w_err = np.abs(pred_w - true_w)
                            l_err = np.abs(pred_l - true_l)

                        defect_area = true_w * true_l
                        # 计算距离探测板边界的最小距离
                        dist_x = MAX_SURF_XY - np.abs(true_cx) - (true_w / 2.0)
                        dist_y = MAX_SURF_XY - np.abs(true_cy) - (true_l / 2.0)
                        dist_bound = min(dist_x, dist_y)

                        # 登记入库
                        sample_stats[physical_id].update({
                            'sample_id': physical_id,
                            'cx': true_cx, 'cy': true_cy, 'w': true_w, 'l': true_l,
                            'heat_iou': iou,
                            'field_rmse': 0.0,  # 占位，循环结束后算
                            'box_center_error': c_err,
                            'box_w_error': w_err,
                            'box_l_error': l_err,
                            'defect_area': defect_area,
                            'distance_to_boundary': dist_bound,
                            # 暂存画图需要的数据
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

        # =====================================================================
        # 数据大汇总与 CSV 导出
        # =====================================================================
        records = []
        for pid, stats in sample_stats.items():
            stats['field_rmse'] = np.sqrt(stats['field_mse_sum'] / stats['excite_count'])
            records.append(stats)

        df = pd.DataFrame(records)

        # 丢弃不可序列化的 plot_data 列，保存纯净数据
        df_export = df.drop(columns=['plot_data', 'field_mse_sum', 'excite_count'])
        csv_name = f"Report_{dataset_name}.csv"
        df_export.to_csv(csv_name, index=False)

        print(f"\n📄 [{dataset_name}] 核心分析报告已生成: {csv_name}")

        # 🚀 专家要求的高级统计量：评估长尾分布与鲁棒性
        if not df.empty:
            rmse_arr = df['field_rmse'].values
            iou_mean = df['heat_iou'].mean() * 100

            rmse_mean = rmse_arr.mean()
            rmse_median = np.median(rmse_arr)
            rmse_p90 = np.percentile(rmse_arr, 90)

            # 提取误差最大的前 10 个样本计算平均
            worst_10_rmse = np.sort(rmse_arr)[-10:].mean() if len(rmse_arr) >= 10 else rmse_mean

            print(f"📊 综合指标评估:")
            print(f"   ➤ Test Heat IoU   : {iou_mean:.2f}%")
            print(f"   ➤ Mean RMSE       : {rmse_mean:.4e}")
            print(f"   ➤ Median RMSE     : {rmse_median:.4e}")
            print(f"   ➤ P90 RMSE        : {rmse_p90:.4e} (覆盖90%工况的误差上界)")
            print(f"   ➤ Worst-10 RMSE   : {worst_10_rmse:.4e} 🚨 (极端最差工况表现)")

    # =====================================================================
    # 继续为你提供优雅的画图功能 (由 DataFrame 驱动排序)
    # =====================================================================
    if plot_top_worst and not df.empty:
        # 去掉完全没预测出框的 NaN 数据后再排序
        df_sorted = df.dropna(subset=['box_center_error']).sort_values('heat_iou', ascending=False)

        if len(df_sorted) >= 3:
            best_3 = df_sorted.head(3).to_dict('records')
            worst_3 = df_sorted.tail(3).iloc[::-1].to_dict('records')

            for rank, data in enumerate(best_3):
                plot_visualizations(data['plot_data'],
                                    f"{dataset_name}_Best_Top{rank + 1}_(IoU_{data['heat_iou'] * 100:.1f}%)")

            for rank, data in enumerate(worst_3):
                plot_visualizations(data['plot_data'],
                                    f"{dataset_name}_Worst_Top{rank + 1}_(IoU_{data['heat_iou'] * 100:.1f}%)")


def plot_visualizations(data, title_prefix):
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(title_prefix, fontsize=16, fontweight='bold')

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

    plt.savefig(f"{title_prefix}.jpg", dpi=300)
    plt.close()


def main():
    print("🔬 启动 SCF-DeepONet 终极量化评估脚本...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SCFDeepONet_V1().to(device)
    ckpt_path = "checkpoints/best_stage2_rmse.pth"
    print(f"📦 正在加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✅ 成功加载第 {checkpoint['epoch']} 轮保存的模型！")

    h5_path_main = 'EMT_SCF_Dataset.h5'
    val_ids = np.load("splits/val_ids.npy")
    evaluate_dataset(model, "Validation_Set", h5_path_main, val_ids, device, plot_top_worst=False)

    h5_path_test = 'EMT_SCF_Test_Dataset.h5'
    if os.path.exists(h5_path_test):
        with h5py.File(h5_path_test, 'r') as hf:
            total_test_samples = hf['v_pair'].shape[0]
        test_ids = np.arange(total_test_samples)
        evaluate_dataset(model, "Report_A_PureData", h5_path_test, test_ids, device, plot_top_worst=True)
    else:
        print("\n⚠️ 找不到 EMT_SCF_Test_Dataset.h5！")


if __name__ == '__main__':
    main()