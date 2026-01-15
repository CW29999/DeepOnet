from scipy.interpolate import griddata
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
class Get_dataset(Dataset):
    def __init__(self, epsilon_data, x_data, ez_data, norm_):
        self.epsilon_data = epsilon_data  # 3D tensor (B, 64, 64)
        self.x_data = torch.as_tensor(x_data, dtype=torch.float32)  # 2D tensor (4096, 2)
        self.ez_data = torch.as_tensor(ez_data, dtype=torch.float32)  # 3D tensor (B, 4096, 2)
        self.norm_stats = norm_
        
        # Create grid for interpolation
        self.grid_x, self.grid_y = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))

    def interpolate_epsilon(self, epsilon, xy_coords):
        """Interpolate epsilon values onto the points defined by xy_coords"""
        epsilon_flat = epsilon.flatten()
        xy_grid = np.array(list(zip(self.grid_x.flatten(), self.grid_y.flatten())))

        # Interpolate to match X_train's 4096 points
        epsilon_interpolated = griddata(xy_grid, epsilon_flat, xy_coords, method='linear')
        return torch.tensor(epsilon_interpolated, dtype=torch.float32)
    
    def __getitem__(self, index):
        epsilon = self.epsilon_data[index]  # Shape (64, 64) - 原始epsilon用于CNN
        x = self.x_data  # Shape (4096, 2) - 坐标数据
        ez_real = self.ez_data[index, :, 0]  # Shape (4096,)
        ez_imag = self.ez_data[index, :, 1]  # Shape (4096,)

        # 归一化epsilon (64, 64) - 直接对原始epsilon进行归一化
        epsilon_norm = (epsilon - self.norm_stats['Epsilon_mean']) / self.norm_stats['Epsilon_std']
        
        # 归一化坐标 (4096, 2)
        x_norm = (x[:, 0] - self.norm_stats['X_mean']) / self.norm_stats['X_std']
        y_norm = (x[:, 1] - self.norm_stats['Y_mean']) / self.norm_stats['Y_std']
        
        # 归一化Ez标签 (4096, 2)
        ez_real = (ez_real - self.norm_stats['Ez_real_mean']) / self.norm_stats['Ez_real_std']
        ez_imag = (ez_imag - self.norm_stats['Ez_imag_mean']) / self.norm_stats['Ez_imag_std']
        
        # 确保epsilon是torch tensor
        if not isinstance(epsilon_norm, torch.Tensor):
            epsilon_norm = torch.as_tensor(epsilon_norm, dtype=torch.float32)
        
        # 添加channel维度用于CNN: (64, 64) -> (1, 64, 64)
        epsilon_norm = epsilon_norm.unsqueeze(0)  # Shape (1, 64, 64)
        
        # Create tensors
        coords = torch.stack([x_norm, y_norm], dim=-1)  # Shape (4096, 2)
        ez = torch.stack([ez_real, ez_imag], dim=-1)  # Shape (4096, 2)
        
        return epsilon_norm, coords, ez

    def __len__(self):
        return len(self.epsilon_data)

# Usage in main
if __name__ == "__main__":
    # Load the data
    data_set = loadmat('pi_MLP_DON/cnn_test/deepOnet_scat_data_re_imag_cnn.mat')
    
    Epsilon_train = data_set['Eplison_train']  # (390, 64, 64)
    X_train = data_set['X_train']  # (4096, 2)
    Ez_train = data_set['realEz_train']  # (390, 4096, 2)
    Epsilon_test = data_set['Eplison_test']  # (98, 64, 64)
    X_test = data_set['X_test']  # (4096, 2)
    Ez_test = data_set['realEz_test']  # (98, 4096, 2)

    # Get the number of batches
    B, M, _ = Epsilon_train.shape
    print(f'Epsilon_train shape: {Epsilon_train.shape}, X_train shape: {X_train.shape}, Ez_train shape: {Ez_train.shape}')

    # Prepare dataset
    norm_stats = {}
    norm_stats['Epsilon_mean'] = float(np.mean(Epsilon_train))
    norm_stats['Epsilon_std'] = float(np.std(Epsilon_train))
    
    # 计算X坐标的均值和标准差
    norm_stats['X_mean'] = float(np.mean(X_train[:, 0]))
    norm_stats['X_std'] = float(np.std(X_train[:, 0]))
    
    # 计算Y坐标的均值和标准差
    norm_stats['Y_mean'] = float(np.mean(X_train[:, 1]))
    norm_stats['Y_std'] = float(np.std(X_train[:, 1]))
    
    # 计算Ez实部的均值和标准差 (对所有样本的所有4096个点的实部)
    norm_stats['Ez_real_mean'] = float(np.mean(Ez_train[:, :, 0]))
    norm_stats['Ez_real_std'] = float(np.std(Ez_train[:, :, 0]))
    
    # 计算Ez虚部的均值和标准差 (对所有样本的所有4096个点的虚部)
    norm_stats['Ez_imag_mean'] = float(np.mean(Ez_train[:, :, 1]))
    norm_stats['Ez_imag_std'] = float(np.std(Ez_train[:, :, 1]))
    
    # 打印归一化统计量（可选，用于调试）
    print("归一化统计量:")
    print(f"Epsilon: mean={norm_stats['Epsilon_mean']:.6f}, std={norm_stats['Epsilon_std']:.6f}")
    print(f"X: mean={norm_stats['X_mean']:.6f}, std={norm_stats['X_std']:.6f}")
    print(f"Y: mean={norm_stats['Y_mean']:.6f}, std={norm_stats['Y_std']:.6f}")
    print(f"Ez_real: mean={norm_stats['Ez_real_mean']:.6f}, std={norm_stats['Ez_real_std']:.6f}")
    print(f"Ez_imag: mean={norm_stats['Ez_imag_mean']:.6f}, std={norm_stats['Ez_imag_std']:.6f}")
    # ... (code to calculate means and stds for normalization)
    
    # Prepare the dataset instances
    Train_dataset = Get_dataset(Epsilon_train, X_train, Ez_train, norm_stats)
    Test_dataset = Get_dataset(Epsilon_test, X_test, Ez_test, norm_stats)

    print(f'Number of samples in Train dataset: {len(Train_dataset)}, Test dataset: {len(Test_dataset)}')

    # Example DataLoader
    loader = DataLoader(Train_dataset, batch_size=4, shuffle=True)
    for B_eps, T_xy, Ez in loader:
        print(f'B_eps shape: {B_eps.shape}, T_xy shape: {T_xy.shape}, Ez shape: {Ez.shape}')
        break
