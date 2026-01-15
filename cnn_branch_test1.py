import torch
# import modules
from tqdm.auto import tqdm  
import numpy as np
from getdata import Get_dataset
# deep learning modules
import torch
from scipy.io import loadmat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import datetime
import pandas as pd
# Plot modules
import matplotlib.pyplot as plt
from scipy.interpolate import griddata  
from pathlib import Path
from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Modified_MLP_Block(nn.Module):
    def __init__(self, input_dim, hidden_channel, output_dim, hidden_size=4):
        super(Modified_MLP_Block, self).__init__()
        self.activation = nn.Tanh()
        self.encodeU = nn.Linear(input_dim, hidden_channel)
        self.encodeV = nn.Linear(input_dim, hidden_channel)
        self.In = nn.Linear(input_dim, hidden_channel)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channel, hidden_channel) for _ in range(hidden_size)
        ])
        self.out = nn.Linear(hidden_channel, output_dim)
        self._init_weights()

    def _init_weights(self):
        g = torch.Generator()
        g.manual_seed(123)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1, generator=g)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        U = self.activation(self.encodeU(x))
        V = self.activation(self.encodeV(x))
        Hidden = self.activation(self.In(x))

        for layer in self.hidden_layers:
            Z = self.activation(layer(Hidden))
            Hidden = (1 - Z) * U + Z * V

        x = self.out(Hidden)
        return x
    


class CNN_Branch_Residual(nn.Module):
    """带残差连接的CNN分支模型"""
    
    def __init__(self, in_channels=1, num_classes=128):
        super(CNN_Branch_Residual, self).__init__()
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 下采样连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim):
        super(DeepONet, self).__init__()
        self.output_dim = output_dim
        self.branch_net = CNN_Branch_Residual(in_channels=branch_input_dim, num_classes=output_dim)
        self.trunk_net = Modified_MLP_Block(trunk_input_dim, hidden_channel, output_dim)

    def forward(self, branch_input, trunk_input):
        
        branch_out = self.branch_net(branch_input)
        trunk_out = self.trunk_net(trunk_input)

        B1, B2 = branch_out[:, :self.output_dim//2], branch_out[:, self.output_dim//2:]
        T1, T2 = trunk_out[:, :self.output_dim//2], trunk_out[:, self.output_dim//2:]
        #print("B1 shape:", B1.shape, "B2 shape:", B2.shape)
        #print("T1 shape:", T1.shape, "T2 shape:", T2.shape)
        s_re = torch.einsum('bi,ni->bn', B1, T1) #实部
        s_im = torch.einsum('bi,ni->bn', B2, T2)
        return s_re, s_im
    
def calculate_derivative(y, x) :
        return torch.autograd.grad(y, x, create_graph=True,\
                        grad_outputs=torch.ones(y.size()).to(device))[0]
    
class PINN_maxwell():
    def __init__(self, model, batch_size = 100, learning_rate = 1e-3, step_size = 200, gamma = 0.85, matpath = 'deepOnet_scat_data_re_imag_1600.mat'):
        self.mu = 1
        self.lam = 1
        self.k0 = 2 * np.pi / self.lam

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_log = [['total_loss', 'pde_loss', 'data_loss']]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.matpath = matpath
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.losses = []
        self.lamda =  []
        self.file_name = 'model_data_driven.pth'
        self.train_set, self.test_set, self.norm_stats = self.load_dataset()
        self.train_loader = DataLoader(self.train_set, batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size = len(self.test_set), shuffle=False)
        

    def load_model(self):
        self.model.load_state_dict(torch.load(self.file_name, weights_only=True))

    def E_function(self, epsilon_data, coord_data):
        epsilon_data = epsilon_data.to(self.device).requires_grad_()
        coord_data = coord_data.to(self.device).requires_grad_()
        return self.model(epsilon_data, coord_data)
    
    def get_data_loss(self, epsilon_data, coord_data, E_true):
        E_re_pred, E_im_pred = self.E_function(epsilon_data, coord_data)
        E_re_true = E_true[:,:, 0]
        E_im_true = E_true[:,:, 1]
        data_loss = self.loss_fn(E_re_pred, E_re_true) + self.loss_fn(E_im_pred, E_im_true)
        return data_loss
    
    
    @torch.no_grad()
    def test_E_loss(self):
        self.model.eval()
        test_loss = 0.0
        for epsilon_data, coord_data, E_true in self.test_loader:
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data[0,:].to(self.device)  # 使用第一个样本的坐标（所有样本共享）
            E_true = E_true.to(self.device)
        
            E_re_pred, E_im_pred = self.E_function(epsilon_data, coord_data)
            # 修正：E_true的形状是(batch_size, 4096, 2)
            # 应该使用[:, :, 0]提取实部，[:, :, 1]提取虚部
            E_re_true = E_true[:, :, 0]  # Shape: (batch_size, 4096)
            E_im_true = E_true[:, :, 1]  # Shape: (batch_size, 4096)

            test_loss += self.loss_fn(E_re_pred, E_re_true) + self.loss_fn(E_im_pred, E_im_true)
        self.model.train()    
        return test_loss / len(self.test_loader)
    
    def train(self, epochs):
        self.losses.append(['epoch',  'data_loss', 'test_loss'])
        start_time = datetime.datetime.now()
        for epoch in tqdm(range(epochs), desc='Training'):
            self.model.train()
            total_loss = 0.0
            data_loss = 0.0
            
            for epsilon_data, coord_data, E_true in self.train_loader:

                epsilon_data = epsilon_data.to(self.device)
                coord_data = coord_data[0,:].to(self.device)
                E_true = E_true.to(self.device)

                self.optimizer.zero_grad()
                
                
                data_loss = self.get_data_loss(epsilon_data, coord_data, E_true)
                loss = data_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_total_loss = total_loss / len(self.train_loader)
            
            avg_data_loss = data_loss / len(self.train_loader)
            avg_test_loss = self.test_E_loss()
            #print(len(self.train_loader))
            self.losses.append([epoch, avg_total_loss, avg_data_loss.item(), avg_test_loss.item()])
            self.scheduler.step()
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, data Loss: {avg_data_loss}, test Loss {avg_test_loss}')
        torch.save(self.model.state_dict(), self.file_name)
        print("Current learning rate:", self.optimizer.param_groups[0]['lr'])
        print("Training Time:",(datetime.datetime.now() - start_time).total_seconds(), "s")

    def plot_loss(self):
        data = np.array(self.losses[1:])
        epochs = data[:, 0]
        train_loss = data[:, 1]
        test_loss = data[:, 2]
        plt.figure(figsize=(10, 6))
        plt.title('Training/Test Loss')
        plt.semilogy(epochs, train_loss, label='train_loss')
        plt.semilogy(epochs, test_loss,  label='test_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        save_path = 'results/loss_plot.png'
        plt.savefig(save_path)
        plt.show()


    def predict(self, epsilon_data, coord_data):
        epsilon_data = torch.tensor(epsilon_data, dtype=torch.float32, requires_grad=True)
        coord_data = torch.tensor(coord_data, dtype=torch.float32, requires_grad=True)
        E_re_pred, E_im_pred = self.E_function(epsilon_data, coord_data)
        E_re_pred = E_re_pred.cpu().detach().numpy()
        E_im_pred = E_im_pred.cpu().detach().numpy()
        return E_re_pred, E_im_pred
    

    def get_predict(self, epsilon_data, coord_data):
        Ere_pred, Eim_pred = self.predict(epsilon_data, coord_data)
        Ere_pred = (Ere_pred * self.norm_stats['Ez_real_std']) + self.norm_stats['Ez_real_mean']
        Eim_pred = (Eim_pred * self.norm_stats['Ez_imag_std']) + self.norm_stats['Ez_imag_mean']
        return Ere_pred, Eim_pred
    
    def get_exact(self):

        for epsilon_data, coord_data, E_test in self.test_loader:
            epsilon_data = epsilon_data.to(self.device)
            coord_data = coord_data[0,:].to(self.device)
            E_test = E_test.to(self.device)
        epsilon_data = (epsilon_data).cpu().detach().numpy() * self.norm_stats['Epsilon_std'] + self.norm_stats['Epsilon_mean']
        coord_data = (coord_data.cpu().detach().numpy() * np.array([self.norm_stats['X_std'], self.norm_stats['Y_std']]) + np.array([self.norm_stats['X_mean'], self.norm_stats['Y_mean']]))
        E_test = (E_test.cpu().detach().numpy() * np.array([self.norm_stats['Ez_real_std'], self.norm_stats['Ez_imag_std']]) + np.array([self.norm_stats['Ez_real_mean'], self.norm_stats['Ez_imag_mean']]))

        return epsilon_data, coord_data, E_test
    
    def load_dataset(self):
        data_set = loadmat('deepOnet_scat_data_re_imag_cnn.mat')
    
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

        return Train_dataset, Test_dataset, norm_stats
    
    def plot_E_re_im(self, idx = 0):
        epsilon_test_exact, coord_test_exact, E_test_exact = self.get_exact()
        for epsilon_data, coord_data, _ in pinn.test_loader:
            Ere_pred, Eim_pred = pinn.get_predict(epsilon_data, coord_data[0,:])
        savemat('results/E_re_im_scat_sbc_data_driven.mat', {'Ere_pred': Ere_pred,'Eim_pred':Eim_pred ,'E_test': E_test_exact, 'Coord_data': coord_test_exact,'epsilon_data': epsilon_test_exact})
        Ere_pred = Ere_pred[idx, :]
        Eim_pred = Eim_pred[idx, :]
        X = coord_test_exact[:, 0]
        Y = coord_test_exact[:, 1]
        Xi, Yi = np.meshgrid(np.unique(X), np.unique(Y))
        re_error = np.abs(Ere_pred - E_test_exact[idx, :, 0])
        im_error = np.abs(Eim_pred - E_test_exact[idx, :, 1])
        print((f'L2 of real part: {np.linalg.norm(re_error)/np.linalg.norm(E_test_exact[idx, :, 0])}, '
               f'L2 of imaginary part: {np.linalg.norm(im_error)/np.linalg.norm(E_test_exact[idx, :, 1])}'))
        def interp(z):
            Zi = griddata(points=(X, Y),   # ← tuple 而非 list，坐标不变
                        values=z,
                        xi=(Xi, Yi),
                        method='cubic',)
            from scipy.ndimage import gaussian_filter
            Zi = gaussian_filter(Zi, sigma=1.0)
            return Zi

        fields = [
            (interp(Ere_pred), 'E_real_pred' , 'bwr'),
            (interp(E_test_exact[idx, :, 0]), 'E_real_true' , 'bwr'),
            (interp(re_error), 'E_real_error', 'bwr'),
            (interp(Eim_pred), 'E_imag_pred' , 'bwr'),
            (interp(E_test_exact[idx, :, 1]), 'E_imag_true' , 'bwr'),
            (interp(im_error), 'E_imag_error', 'bwr'),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
        for ax, (Zi, title, cmap) in zip(axes.ravel(), fields):
            pcm = ax.pcolormesh(Xi, Yi, Zi, shading='gouraud', cmap=cmap)
            ax.set_title(title)
            ax.set_xlabel('X (µm)')
            ax.set_ylabel('Y (µm)')
            ax.set_aspect('equal')
            fig.colorbar(pcm, ax=ax, shrink=0.8)
        plt.tight_layout()
        save_path = 'results/E_re_im_plot.png'
        plt.savefig(save_path)
        plt.show()
        

if __name__ == "__main__":
    epochs = 50000
    batch_size = 390
    step_size = 1000
    model = DeepONet(branch_input_dim=1, trunk_input_dim=2, hidden_channel=128, output_dim=64)
    pinn = PINN_maxwell(model, batch_size=batch_size, learning_rate=1e-3, step_size=step_size, gamma=0.95, matpath ='deepOnet_scat_data_re_imag_cnn.mat')
    pinn.train(epochs=epochs)
    pinn.plot_loss()
    #pinn.load_model()
    pinn.plot_E_re_im(idx=30)



