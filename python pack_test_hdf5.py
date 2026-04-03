import os
import glob
import h5py
import numpy as np
import mat73
import scipy.io as sio


def load_mat_safe(file_path):
    """ 兼容读取旧版 scipy 和新版 mat73 (v7.3) 格式的 .mat 文件 """
    try:
        return sio.loadmat(file_path)
    except NotImplementedError:
        return mat73.loadmat(file_path)


def get_coil_coords():
    coords = np.zeros((16, 2), dtype=np.float32)
    grid_1d = np.array([-24, -8, 8, 24], dtype=np.float32)
    idx = 0
    for y in grid_1d:
        for x in grid_1d:
            coords[idx, 0] = x
            coords[idx, 1] = y
            idx += 1
    return coords


def main():
    print("================ 开始构建 SCF-DeepONet [绝密测试集] ================")

    # 1. 加载基准数据
    base_data = load_mat_safe('../data/Baseline_Data.mat')
    V_b = base_data['V_baseline_fp32']
    A_b = base_data['A_baseline_fp32']
    pts_3d = base_data['Evaluation_Points_fp32']

    # 2. 定位到测试集专属路径
    test_data_dir = r'../test_data'
    defect_files = sorted(glob.glob(os.path.join(test_data_dir, 'test_sample_*.mat')))
    num_samples = len(defect_files)

    if num_samples == 0:
        raise ValueError(f"❌ 警告：在 {test_data_dir} 中没有找到任何文件！请检查路径。")
    print(f"🔍 成功找到 {num_samples} 个绝密测试样本。")

    MAX_SURF_XY = 32.0
    coords_xy_surf_norm = np.clip(pts_3d[:, :2] / MAX_SURF_XY, -1.0, 1.0)
    MAX_FIELD_XYZ = np.array([130.0, 90.0, 90.0], dtype=np.float32)
    coords_3d_field_norm = pts_3d / MAX_FIELD_XYZ

    coil_coords = get_coil_coords()
    geom_features = np.zeros((16, 16, 7), dtype=np.float32)
    for t in range(16):
        for r in range(16):
            xt, yt = coil_coords[t]
            xr, yr = coil_coords[r]
            dx, dy = xr - xt, yr - yt
            d = np.sqrt(dx ** 2 + dy ** 2)
            geom_features[t, r] = np.array([
                xt / 32.0, yt / 32.0, xr / 32.0, yr / 32.0,
                dx / 64.0, dy / 64.0, d / 68.0
            ], dtype=np.float32)

    # =========================================================================
    # 🚀 【核心防坑点】测试集的 A_scale 必须与训练集绝对一致！
    # 绝不能重新计算，否则会导致评估指标产生严重的系统性偏差。
    # =========================================================================
    try:
        with h5py.File('EMT_SCF_Dataset.h5', 'r') as hf_train:
            A_scale = hf_train.attrs['A_scale']
        print(f"\n[Pass 1] 锁定训练集标度: 成功读取训练集 A_scale = {A_scale:.4e}")
    except Exception as e:
        raise RuntimeError(
            "❌ 致命错误：无法读取 EMT_SCF_Dataset.h5！请确保该文件与本脚本在同一目录下。测试集必须继承训练集的 A_scale。")

    print("\n[Pass 2] 正在构建 测试集 HDF5 (继承优化 Chunk 模式)...")
    off_diag_mask = ~np.eye(16, dtype=bool)

    # 3. 输出文件名为专属的 EMT_SCF_Test_Dataset.h5
    with h5py.File('EMT_SCF_Test_Dataset.h5', 'w') as hf:
        hf.create_dataset('coords_xy_surf_norm', data=coords_xy_surf_norm, dtype='float32')
        hf.create_dataset('coords_3d_field_norm', data=coords_3d_field_norm, dtype='float32')

        hf.attrs['A_scale'] = A_scale
        hf.attrs['MAX_SURF_XY'] = MAX_SURF_XY
        hf.attrs['MAX_FIELD_XYZ'] = MAX_FIELD_XYZ

        ds_Vpair = hf.create_dataset('v_pair', shape=(num_samples, 16, 15, 9), dtype='float32', chunks=True,
                                     compression='lzf')
        ds_A = hf.create_dataset('A_delta_norm', shape=(num_samples, 16, 50000, 6), dtype='float32',
                                 chunks=(1, 1, 2048, 6), compression=None)
        ds_A0 = hf.create_dataset('A0_norm', shape=(16, 50000, 6), dtype='float32',
                                  chunks=(1, 2048, 6), compression=None)
        ds_labels = hf.create_dataset('labels_norm', shape=(num_samples, 4), dtype='float32')

        A0_scaled = A_b / A_scale
        A0_feat = np.zeros((16, 50000, 6), dtype=np.float32)
        A0_feat[..., 0:2] = np.stack([A0_scaled[..., 0].real, A0_scaled[..., 0].imag], axis=-1)
        A0_feat[..., 2:4] = np.stack([A0_scaled[..., 1].real, A0_scaled[..., 1].imag], axis=-1)
        A0_feat[..., 4:6] = np.stack([A0_scaled[..., 2].real, A0_scaled[..., 2].imag], axis=-1)
        ds_A0[:] = A0_feat

        for i, file in enumerate(defect_files):
            data = load_mat_safe(file)
            V_d = data['V_defect_fp32']
            A_d = data['A_defect_fp32']
            label_raw = np.array(data['defect_labels']).flatten()

            eps = 1e-6
            V_norm = (V_b - V_d) / (np.abs(V_b) + eps)

            v_pair_sample = np.zeros((16, 15, 9), dtype=np.float32)
            for t in range(16):
                v_norm_15 = V_norm[t, off_diag_mask[t]]
                geom_15 = geom_features[t, off_diag_mask[t]]
                v_pair_sample[t, :, 0] = v_norm_15.real
                v_pair_sample[t, :, 1] = v_norm_15.imag
                v_pair_sample[t, :, 2:] = geom_15

            Delta_A_norm = (A_d - A_b) / A_scale
            A_feat = np.zeros((16, 50000, 6), dtype=np.float32)
            A_feat[..., 0:2] = np.stack([Delta_A_norm[..., 0].real, Delta_A_norm[..., 0].imag], axis=-1)
            A_feat[..., 2:4] = np.stack([Delta_A_norm[..., 1].real, Delta_A_norm[..., 1].imag], axis=-1)
            A_feat[..., 4:6] = np.stack([Delta_A_norm[..., 2].real, Delta_A_norm[..., 2].imag], axis=-1)

            label_norm = label_raw / MAX_SURF_XY

            ds_Vpair[i] = v_pair_sample
            ds_A[i] = A_feat
            ds_labels[i] = label_norm

            if (i + 1) % 10 == 0 or (i + 1) == num_samples:
                print(f"    已写入 {i + 1}/{num_samples} 样本...")

    print("================ 测试数据集打包完美收官！ ================")


if __name__ == '__main__':
    main()