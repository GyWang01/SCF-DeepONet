import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载 MATLAB 生成的点云数据
# 请确保 'Fixed_Evaluation_Points.mat' 与此脚本在同一目录下
try:
    mat_data = sio.loadmat('Fixed_Evaluation_Points.mat')
    pts = mat_data['Evaluation_Points']
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    print(f"成功加载点云数据，共 {pts.shape[0]} 个采样点。")
except FileNotFoundError:
    print("错误：找不到 'Fixed_Evaluation_Points.mat' 文件，请检查路径。")
    exit()

# 2. 设置画布和子图
fig = plt.figure(figsize=(18, 6))
fig.suptitle('PINN EMT 3D Evaluation Points Distribution', fontsize=16, fontweight='bold')

# -----------------------------------------------------------
# 子图 1: 3D 全局视角
# -----------------------------------------------------------
ax1 = fig.add_subplot(131, projection='3d')
# 使用 Z 轴的深度作为颜色映射 (cmap)，直观显示分层
sc1 = ax1.scatter(x, y, z, c=z, cmap='turbo', s=2, alpha=0.6)
ax1.set_title('3D Global View', fontsize=14)
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
# 设置视角，更利于观察下方铁块
ax1.view_init(elev=20, azim=45)

# -----------------------------------------------------------
# 子图 2: X-Z 侧视图 (极其重要：观察趋肤层密度)
# -----------------------------------------------------------
ax2 = fig.add_subplot(132)
sc2 = ax2.scatter(x, z, c=z, cmap='turbo', s=3, alpha=0.5)
ax2.set_title('X-Z Side View (Focus on Skin Depth)', fontsize=14)
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Z (mm)')
# 画出铁块表面 (Z=0) 和缺陷底部 (Z=-4) 的基准线
ax2.axhline(0, color='red', linestyle='--', linewidth=1, label='Iron Surface (Z=0)')
ax2.axhline(-4, color='orange', linestyle='--', linewidth=1, label='Defect Bottom (Z=-4)')
# 限制 Z 轴范围，重点观察铁块内部和近表面空气层
ax2.set_ylim(-6, 2)
ax2.set_xlim(-40, 40)
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)

# -----------------------------------------------------------
# 子图 3: X-Y 俯视图 (敏感区覆盖情况)
# -----------------------------------------------------------
ax3 = fig.add_subplot(133)
sc3 = ax3.scatter(x, y, c=z, cmap='turbo', s=3, alpha=0.5)
ax3.set_title('X-Y Top View (Sensor Array Region)', fontsize=14)
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
# 画出你定义的敏感区域边界 [-32, 32]
rect = plt.Rectangle((-32, -32), 64, 64, fill=False, color='red', linestyle='--', linewidth=1.5, label='Sensitive ROI')
ax3.add_patch(rect)
ax3.axis('equal')  # 保持 X 和 Y 比例一致
ax3.set_xlim(-50, 50)
ax3.set_ylim(-50, 50)
ax3.legend()
ax3.grid(True, linestyle=':', alpha=0.6)

# 添加全局颜色条
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sc1, cax=cbar_ax)
cbar.set_label('Z-Depth (mm)', rotation=270, labelpad=15)

plt.subplots_adjust(left=0.05, right=0.9, wspace=0.2)

plt.savefig('Evaluation_Points_Distribution.png')
plt.show()
