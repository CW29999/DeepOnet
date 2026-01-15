clc; clear;

%% 文件列表
mph_dir = "model_solved";
files = dir(fullfile(mph_dir, "*.mph"));
assert(~isempty(files), "model_solved 里没找到 .mph 文件");

%% 传感器网格（只生成一次）
num_samples = 64*64;
nSide = sqrt(num_samples);
assert(abs(nSide-round(nSide)) < 1e-12, "num_samples 必须是完全平方数");

x = linspace(-5e-6, 5e-6, nSide);
y = linspace(-5e-6, 5e-6, nSide);
[X, Y] = meshgrid(x, y);
xx = [X(:)'; Y(:)'];        % 2 x num_samples
X_sensor = xx.';            % num_samples x 2  (你的 net_y)

%% 读取所有 mph，暂存到 cell
EpsCell = cell(numel(files), 1);     % 每个元素: [nSol_i x num_samples]
EzCell  = cell(numel(files), 1);     % 每个元素: [nSol_i x num_samples x 2]
nPerFile = zeros(numel(files), 1);

for i = 1:numel(files)
    fpath = fullfile(files(i).folder, files(i).name);
    model = mphload(fpath);

    % 读全量（all outer solutions）
    epsAll = mphinterp(model, 'ewfd.epsilonrxx', 'coord', xx, 'dataset', 'dset1', 'outersolnum', 'all');
    ezAll  = mphinterp(model, 'ewfd.Ez',        'coord', xx, 'dataset', 'dset1', 'outersolnum', 'all');

    % epsAll / ezAll 的第一维是 outer solution 数量（样本数）
    nSol = size(epsAll, 1);
    nPerFile(i) = nSol;

    % reshape 成 [nSol x num_samples]
    epsAll = reshape(epsAll, [num_samples, nSol]).';
    ezAll  = reshape(ezAll,  [num_samples, nSol]).';

    % Ez: 实部/虚部 -> 第3维通道
    ezAll = cat(3, real(ezAll), imag(ezAll));  % [nSol x num_samples x 2]

    EpsCell{i} = epsAll;
    EzCell{i}  = ezAll;

    % 释放 COMSOL 对象（可选但建议）
    try, ModelUtil.remove('model'); end %#ok<TRYNC>
    clear model;

    fprintf("Read %d/%d: %s, nSol=%d\n", i, numel(files), files(i).name, nSol);
end

%% 拼接成全量数据
Eplison_all = cat(1, EpsCell{:});   % [Ntotal x num_samples]
Ez_all      = cat(1, EzCell{:});    % [Ntotal x num_samples x 2]
Ntotal = size(Eplison_all, 1);

fprintf("Total samples = %d\n", Ntotal);

%% 划分数据集
pctTrain = 0.8;
idx = randperm(Ntotal);
nTrain = floor(pctTrain * Ntotal);
trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

Eplison_train = Eplison_all(trainIdx, :);
Eplison_test  = Eplison_all(testIdx, :);

Ez_train = Ez_all(trainIdx, :, :);
Ez_test  = Ez_all(testIdx,  :, :);

X_train = X_sensor;
X_test  = X_sensor;

%% 保存（8GB 必须 -v7.3）
out_mat = "deepOnet_scat_data_4096_split.mat";
save(out_mat, ...
    "Eplison_train","Eplison_test","Ez_train","Ez_test", ...
    "X_train","X_test", ...
    "trainIdx","testIdx","nPerFile", ...
    "-v7.3");

fprintf("Saved: %s\n", out_mat);
