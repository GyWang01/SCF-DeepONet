import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def main():
    print("📊 启动 SCF-DeepONet 成对统计分析引擎 (Pairwise Analysis)...")

    # 1. 配置文件路径 (请确保文件名与你实际生成的保持一致)
    file_A = "Report_Report_A_PureData.csv"
    file_B = "Report_Report_B_NoPDE.csv"
    file_C = "Report_Report_C_PDE.csv"

    # 检查文件是否存在
    for f in [file_A, file_B, file_C]:
        if not os.path.exists(f):
            print(f"❌ 找不到文件: {f}，请检查路径或文件名！")
            return

    # 2. 读取数据
    df_A = pd.read_csv(file_A)
    df_B = pd.read_csv(file_B)
    df_C = pd.read_csv(file_C)

    # 提取我们最关心的 RMSE
    rmse_A = df_A[['sample_id', 'field_rmse']].rename(columns={'field_rmse': 'rmse_PureData'})
    rmse_B = df_B[['sample_id', 'field_rmse']].rename(columns={'field_rmse': 'rmse_NoPDE'})
    rmse_C = df_C[['sample_id', 'field_rmse']].rename(columns={'field_rmse': 'rmse_PDE'})

    # 根据 sample_id 进行严格对齐合并
    df_merged = pd.merge(rmse_A, rmse_B, on='sample_id')
    df_merged = pd.merge(df_merged, rmse_C, on='sample_id')

    # =========================================================
    # 专家建议 1: 按样本逐个算 ΔRMSE = RMSE_PDE - RMSE_NoPDE
    # =========================================================
    df_merged['delta_rmse'] = df_merged['rmse_PDE'] - df_merged['rmse_NoPDE']

    total_samples = len(df_merged)
    num_improved = (df_merged['delta_rmse'] < 0).sum()
    num_worsened = (df_merged['delta_rmse'] > 0).sum()
    num_tied = (df_merged['delta_rmse'] == 0).sum()

    mean_drop = df_merged['delta_rmse'].mean()
    median_drop = df_merged['delta_rmse'].median()

    print("\n" + "=" * 50)
    print(" 🎯 专家指标 1：逐样本 ΔRMSE (PDE vs NoPDE)")
    print("=" * 50)
    print(f"➤ 总样本数: {total_samples}")
    print(f"➤ 变好 (PDE更优): {num_improved} 个 ({num_improved / total_samples * 100:.1f}%)")
    print(f"➤ 变差 (PDE更差): {num_worsened} 个 ({num_worsened / total_samples * 100:.1f}%)")
    print(f"➤ 持平: {num_tied} 个")
    print(f"➤ 平均下降 (Mean ΔRMSE): {mean_drop:.4e}")
    print(f"➤ 中位数下降 (Median ΔRMSE): {median_drop:.4e}")

    # =========================================================
    # 专家建议 2: 配对统计检验 (Wilcoxon signed-rank test)
    # =========================================================
    # Wilcoxon 检验要求两组数据具有配对关系
    stat, p_value = stats.wilcoxon(df_merged['rmse_PDE'], df_merged['rmse_NoPDE'], alternative='less')

    print("\n" + "=" * 50)
    print(" 🎯 专家指标 2：Wilcoxon 符号秩检验 (统计显著性)")
    print("=" * 50)
    print(f"➤ Wilcoxon Statistic: {stat}")
    print(f"➤ P-value: {p_value:.4e}")
    if p_value < 0.05:
        print("✅ 结论: P < 0.05，具有统计学显著性！PDE 的提升是极其可靠的，绝非偶然噪声！")
    else:
        print("⚠️ 结论: P >= 0.05，统计学显著性不足。")

    # =========================================================
    # 专家建议 3: 画 RMSE_NoPDE vs RMSE_PDE 散点图
    # =========================================================
    print("\n" + "=" * 50)
    print(" 🎯 专家指标 3：正在绘制并保存对比散点图...")

    plt.figure(figsize=(8, 8))

    x = df_merged['rmse_NoPDE'].values
    y = df_merged['rmse_PDE'].values

    # 确定坐标轴范围，保证 y=x 线在严格的对角线
    min_val = min(x.min(), y.min()) * 0.95
    max_val = max(x.max(), y.max()) * 1.05

    # 画出散点 (根据改善与否标记颜色)
    improved_mask = y < x
    worsened_mask = y >= x

    plt.scatter(x[improved_mask], y[improved_mask], alpha=0.7, c='green', s=30, label=f'PDE Improved ({num_improved})')
    plt.scatter(x[worsened_mask], y[worsened_mask], alpha=0.7, c='red', s=30, label=f'PDE Worsened ({num_worsened})')

    # 画 y=x 的基准线
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y = x (No Improvement)')

    plt.xlim([min_val, max_val])
    plt.ylim([min_val, max_val])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title('Pairwise Comparison: RMSE_NoPDE vs RMSE_PDE', fontsize=14, fontweight='bold')
    plt.xlabel('RMSE (No PDE Baseline)', fontsize=12)
    plt.ylabel('RMSE (With PDE Regularization)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)

    out_img = 'Scatter_NoPDE_vs_PDE.jpg'
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {out_img}")
    print("=" * 50)


if __name__ == '__main__':
    main()