import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.io import savemat

# 你的数据读取（保持不变）
from getdata import Get_dataset, load_mat73


# =========================
# DDP helpers
# =========================
def ddp_setup():
    """
    torchrun 会自动设置环境变量：
    - RANK, WORLD_SIZE, LOCAL_RANK
    """
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process():
    return dist.get_rank() == 0


def ddp_cleanup():
    dist.destroy_process_group()


# =========================
# Model definitions
# =========================
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


class CNN_Branch_Residual(nn.Module):
    """带残差连接的CNN分支模型"""
    def __init__(self, in_channels=1, num_classes=128):
        super(CNN_Branch_Residual, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 128, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DeepONet(nn.Module):
    """
    这里保留你原来的两输入 forward: (epsilon_data, coord_data)
    DDP 不会像 DataParallel 那样自动把 coord_data 切成 586/580。
    """
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_channel, output_dim):
        super(DeepONet, self).__init__()
        self.output_dim = output_dim
        self.branch_net = CNN_Branch_Residual(in_channels=branch_input_dim, num_classes=output_dim)
        self.trunk_net = Modified_MLP_Block(trunk_input_dim, hidden_channel, output_dim)

    def forward(self, branch_input, trunk_input):
        # branch_input: (B,1,64,64)
        # trunk_input:  (N,2)  N=4096
        branch_out = self.branch_net(branch_input)   # (B, outdim)
        trunk_out = self.trunk_net(trunk_input)      # (N, outdim)

        B1, B2 = branch_out[:, :self.output_dim // 2], branch_out[:, self.output_dim // 2:]
        T1, T2 = trunk_out[:, :self.output_dim // 2], trunk_out[:, self.output_dim // 2:]

        s_re = torch.einsum('bi,ni->bn', B1, T1)  # (B, N)
        s_im = torch.einsum('bi,ni->bn', B2, T2)  # (B, N)
        return s_re, s_im


# =========================
# PINN wrapper
# =========================
class PINN_maxwell:
    def __init__(self,
                 model,
                 device,
                 batch_size=100,
                 learning_rate=1e-3,
                 step_size=200,
                 gamma=0.85,
                 matpath='deepOnet_scat_data_4096_split.mat'):

        self.mu = 1
        self.lam = 1
        self.k0 = 2 * np.pi / self.lam

        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.matpath = matpath

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        self.losses = []
        self.file_name = 'model_data_driven.pth'

        self.train_set, self.test_set, self.norm_stats = self.load_dataset()

        # DDP sampler
        self.train_sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=False)
        self.test_sampler = DistributedSampler(self.test_set, shuffle=False, drop_last=False)

        # DataLoader：注意 sampler 存在时，不要 shuffle=True
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        self.test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

    def load_model(self):
        # DDP 包装下，真实网络在 .module
        state = torch.load(self.file_name, map_location="cpu")
        self.model.module.load_state_dict(state)

    def E_function(self, epsilon_data, coord_data):
        # epsilon_data: (B,1,64,64)
        # coord_data:   (4096,2)
        epsilon_data = epsilon_data.to(self.device, non_blocking=True)
        coord_data = coord_data.to(self.device, non_blocking=True)
        return self.model(epsilon_data, coord_data)

    def get_data_loss(self, epsilon_data, coord_data, E_true):
        E_re_pred, E_im_pred = self.E_function(epsilon_data, coord_data)
        E_re_true = E_true[:, :, 0]
        E_im_true = E_true[:, :, 1]
        return self.loss_fn(E_re_pred, E_re_true) + self.loss_fn(E_im_pred, E_im_true)

    @torch.no_grad()
    def test_E_loss(self):
        self.model.eval()

        # 用 all_reduce 汇总所有 rank 的 loss（按样本数加权）
        loss_sum = torch.zeros(1, device=self.device)
        n_sum = torch.zeros(1, device=self.device)

        for epsilon_data, coord_data, E_true in self.test_loader:
            bs = epsilon_data.size(0)

            epsilon_data = epsilon_data.to(self.device, non_blocking=True)
            # 你的数据坐标对所有样本相同，所以取 batch 的第 0 个坐标即可
            coord_data = coord_data[0, :].to(self.device, non_blocking=True)  # (4096,2)
            E_true = E_true.to(self.device, non_blocking=True)

            E_re_pred, E_im_pred = self.E_function(epsilon_data, coord_data)
            E_re_true = E_true[:, :, 0]
            E_im_true = E_true[:, :, 1]
            batch_loss = self.loss_fn(E_re_pred, E_re_true) + self.loss_fn(E_im_pred, E_im_true)

            loss_sum += batch_loss * bs
            n_sum += bs

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_sum, op=dist.ReduceOp.SUM)

        self.model.train()
        return (loss_sum / n_sum).item()

    def train(self, epochs):
        if is_main_process():
            self.losses.append(['epoch', 'train_loss', 'test_loss'])

        start_time = datetime.datetime.now()

        # tqdm 只在 rank0 展示
        for epoch in tqdm(range(epochs), desc='Training', disable=(not is_main_process())):
            self.model.train()
            self.train_sampler.set_epoch(epoch)

            train_loss_sum = 0.0
            n_batch = 0

            for epsilon_data, coord_data, E_true in self.train_loader:
                epsilon_data = epsilon_data.to(self.device, non_blocking=True)
                coord_data = coord_data[0, :].to(self.device, non_blocking=True)  # (4096,2)
                E_true = E_true.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                loss = self.get_data_loss(epsilon_data, coord_data, E_true)
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item()
                n_batch += 1

            avg_train_loss = train_loss_sum / max(n_batch, 1)
            avg_test_loss = self.test_E_loss()

            if is_main_process():
                self.losses.append([epoch, avg_train_loss, avg_test_loss])
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, train Loss: {avg_train_loss:.6e}, test Loss: {avg_test_loss:.6e}')

            self.scheduler.step()

        # 只在 rank0 保存
        if is_main_process():
            torch.save(self.model.module.state_dict(), self.file_name)
            print("Current learning rate:", self.optimizer.param_groups[0]['lr'])
            print("Training Time:", (datetime.datetime.now() - start_time).total_seconds(), "s")

    def plot_loss(self):
        # 只允许主进程调用
        if not is_main_process():
            return

        data = np.array(self.losses[1:], dtype=object)
        epochs = data[:, 0].astype(int)
        train_loss = data[:, 1].astype(float)
        test_loss = data[:, 2].astype(float)

        plt.figure(figsize=(10, 6))
        plt.title('Training/Test Loss')
        plt.semilogy(epochs, train_loss, label='train_loss')
        plt.semilogy(epochs, test_loss, label='test_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        save_path = 'results/loss_plot.png'
        os.makedirs('results', exist_ok=True)
        plt.savefig(save_path)
        plt.show()

    def predict(self, epsilon_data, coord_data):
        # 预测只建议主进程用
        epsilon_data = torch.tensor(epsilon_data, dtype=torch.float32).to(self.device)
        coord_data = torch.tensor(coord_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            E_re_pred, E_im_pred = self.model(epsilon_data, coord_data)
        return E_re_pred.cpu().numpy(), E_im_pred.cpu().numpy()

    def get_predict(self, epsilon_data, coord_data):
        Ere_pred, Eim_pred = self.predict(epsilon_data, coord_data)
        Ere_pred = (Ere_pred * self.norm_stats['Ez_real_std']) + self.norm_stats['Ez_real_mean']
        Eim_pred = (Eim_pred * self.norm_stats['Ez_imag_std']) + self.norm_stats['Ez_imag_mean']
        return Ere_pred, Eim_pred

    def get_exact(self):
        # 只在主进程调用
        if not is_main_process():
            return None, None, None

        # 取一个 batch（test_sampler 下，每个 rank 拿到的不一样；这里只主进程做）
        epsilon_data, coord_data, E_test = next(iter(self.test_loader))
        epsilon_data = epsilon_data.to(self.device)
        coord_data = coord_data[0, :].to(self.device)
        E_test = E_test.to(self.device)

        epsilon_data = (epsilon_data.cpu().numpy() * self.norm_stats['Epsilon_std'] + self.norm_stats['Epsilon_mean'])
        coord_data = (coord_data.cpu().numpy() * np.array([self.norm_stats['X_std'], self.norm_stats['Y_std']])
                      + np.array([self.norm_stats['X_mean'], self.norm_stats['Y_mean']]))
        E_test = (E_test.cpu().numpy() * np.array([self.norm_stats['Ez_real_std'], self.norm_stats['Ez_imag_std']])
                  + np.array([self.norm_stats['Ez_real_mean'], self.norm_stats['Ez_imag_mean']]))
        return epsilon_data, coord_data, E_test

    def load_dataset(self):
        mat_path = self.matpath

        Epsilon_train = load_mat73(mat_path, 'Eplison_train')
        X_train = load_mat73(mat_path, 'X_train') * 1e6
        Ez_train = load_mat73(mat_path, 'Ez_train')

        Epsilon_test = load_mat73(mat_path, 'Eplison_test')
        X_test = load_mat73(mat_path, 'X_test') * 1e6
        Ez_test = load_mat73(mat_path, 'Ez_test')

        # 如果 Epsilon 是 (n,4096) 这里 reshape 成 (n,64,64)
        if Epsilon_train.ndim == 2 and Epsilon_train.shape[1] == 4096:
            Epsilon_train = Epsilon_train.reshape(Epsilon_train.shape[0], 64, 64)
        if Epsilon_test.ndim == 2 and Epsilon_test.shape[1] == 4096:
            Epsilon_test = Epsilon_test.reshape(Epsilon_test.shape[0], 64, 64)

        # 打印一次（只在主进程）
        if is_main_process():
            print(f'Epsilon_train shape: {Epsilon_train.shape}, X_train shape: {X_train.shape}, Ez_train shape: {Ez_train.shape}')
            print(f'Epsilon_test  shape: {Epsilon_test.shape}, X_test  shape: {X_test.shape}, Ez_test  shape: {Ez_test.shape}')

        # normalization stats（用 train）
        norm_stats = {}
        norm_stats['Epsilon_mean'] = float(np.mean(Epsilon_train))
        norm_stats['Epsilon_std'] = float(np.std(Epsilon_train))
        norm_stats['X_mean'] = float(np.mean(X_train[:, 0]))
        norm_stats['X_std'] = float(np.std(X_train[:, 0]))
        norm_stats['Y_mean'] = float(np.mean(X_train[:, 1]))
        norm_stats['Y_std'] = float(np.std(X_train[:, 1]))
        norm_stats['Ez_real_mean'] = float(np.mean(Ez_train[:, :, 0]))
        norm_stats['Ez_real_std'] = float(np.std(Ez_train[:, :, 0]))
        norm_stats['Ez_imag_mean'] = float(np.mean(Ez_train[:, :, 1]))
        norm_stats['Ez_imag_std'] = float(np.std(Ez_train[:, :, 1]))

        if is_main_process():
            print("归一化统计量:")
            print(f"Epsilon: mean={norm_stats['Epsilon_mean']:.6f}, std={norm_stats['Epsilon_std']:.6f}")
            print(f"X: mean={norm_stats['X_mean']:.6f}, std={norm_stats['X_std']:.6f}")
            print(f"Y: mean={norm_stats['Y_mean']:.6f}, std={norm_stats['Y_std']:.6f}")
            print(f"Ez_real: mean={norm_stats['Ez_real_mean']:.6f}, std={norm_stats['Ez_real_std']:.6f}")
            print(f"Ez_imag: mean={norm_stats['Ez_imag_mean']:.6f}, std={norm_stats['Ez_imag_std']:.6f}")

        Train_dataset = Get_dataset(Epsilon_train, X_train, Ez_train, norm_stats)
        Test_dataset = Get_dataset(Epsilon_test, X_test, Ez_test, norm_stats)
        return Train_dataset, Test_dataset, norm_stats

    def plot_E_re_im(self, idx=0):
        # 只在主进程做可视化/保存
        if not is_main_process():
            return

        epsilon_test_exact, coord_test_exact, E_test_exact = self.get_exact()
        if epsilon_test_exact is None:
            return

        # 用主进程的模型预测
        # 注意：这里用一个 batch 的 epsilon 做预测演示
        epsilon_data, coord_data, _ = next(iter(self.test_loader))
        Ere_pred, Eim_pred = self.get_predict(epsilon_data, coord_data[0, :])

        os.makedirs('results', exist_ok=True)
        savemat('results/E_re_im_scat_sbc_data_driven.mat', {
            'Ere_pred': Ere_pred,
            'Eim_pred': Eim_pred,
            'E_test': E_test_exact,
            'Coord_data': coord_test_exact,
            'epsilon_data': epsilon_test_exact
        })

        Ere_pred = Ere_pred[idx, :]
        Eim_pred = Eim_pred[idx, :]

        X = coord_test_exact[:, 0]
        Y = coord_test_exact[:, 1]
        Xi, Yi = np.meshgrid(np.unique(X), np.unique(Y))

        re_error = np.abs(Ere_pred - E_test_exact[idx, :, 0])
        im_error = np.abs(Eim_pred - E_test_exact[idx, :, 1])

        print((f'L2 of real part: {np.linalg.norm(re_error) / np.linalg.norm(E_test_exact[idx, :, 0])}, '
               f'L2 of imaginary part: {np.linalg.norm(im_error) / np.linalg.norm(E_test_exact[idx, :, 1])}'))

        def interp(z):
            Zi = griddata(points=(X, Y), values=z, xi=(Xi, Yi), method='cubic')
            from scipy.ndimage import gaussian_filter
            Zi = gaussian_filter(Zi, sigma=1.0)
            return Zi

        fields = [
            (interp(Ere_pred), 'E_real_pred'),
            (interp(E_test_exact[idx, :, 0]), 'E_real_true'),
            (interp(re_error), 'E_real_error'),
            (interp(Eim_pred), 'E_imag_pred'),
            (interp(E_test_exact[idx, :, 1]), 'E_imag_true'),
            (interp(im_error), 'E_imag_error'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
        for ax, (Zi, title) in zip(axes.ravel(), fields):
            pcm = ax.pcolormesh(Xi, Yi, Zi, shading='gouraud')
            ax.set_title(title)
            ax.set_xlabel('X (µm)')
            ax.set_ylabel('Y (µm)')
            ax.set_aspect('equal')
            fig.colorbar(pcm, ax=ax, shrink=0.8)

        plt.tight_layout()
        save_path = 'results/E_re_im_plot.png'
        plt.savefig(save_path)
        plt.show()


# =========================
# Main
# =========================
if __name__ == "__main__":
    # DDP init
    local_rank = ddp_setup()
    device = torch.device("cuda", local_rank)

    epochs = 50000
    batch_size = 320
    step_size = 1000

    # build model and wrap DDP
    base_model = DeepONet(branch_input_dim=1, trunk_input_dim=2, hidden_channel=128, output_dim=64).to(device)
    model = DDP(base_model, device_ids=[local_rank], output_device=local_rank)

    pinn = PINN_maxwell(
        model=model,
        device=device,
        batch_size=batch_size,
        learning_rate=1e-3,
        step_size=step_size,
        gamma=0.95,
        matpath='deepOnet_scat_data_4096_split.mat'
    )

    pinn.train(epochs=epochs)

    if is_main_process():
        pinn.plot_loss()
        pinn.plot_E_re_im(idx=26)

    ddp_cleanup()
