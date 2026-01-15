clc; clear; close all;

%% =====================================================
%  用户参数区（只需要改这里）
%% =====================================================

N_models   = 100;        % 生成模型数量
N_scatter  = 3;         % <<< 严格要求的散射体个数
outdir     = 'models_Scatter_3_gen';  % 输出目录

% 随机场参数
N = 128;                % 网格分辨率
sigma = 12;             % Gaussian blur (pixels)
threshold = 0.5;        % 二值阈值
pad = 3*sigma;
min_area_px = 200;       % 最小连通域面积（像素）

% 几何尺寸
L_domain  = 10e-6;       % 计算域边长（中心在原点）
L_scatter = 8e-6;       % 散射体生成区域（中心区域）

max_attempt = 200;      % 单个模型最大尝试次数（防死循环）

%% =====================================================
%  初始化
%% =====================================================

if ~exist(outdir,'dir')
    mkdir(outdir);
end

import com.comsol.model.*
import com.comsol.model.util.*

try
    ModelUtil.showProgress(true);
catch
    mphstart
end

dx = L_scatter / N;

%% =====================================================
%  主循环：批量生成模型
%% =====================================================

for mid = 1:N_models

    fprintf('\n--- Generating model %d / %d ---\n', mid, N_models);

    success = false;
    attempt = 0;

    %% =================================================
    %  拒绝采样：直到散射体个数 == N_scatter
    %% =================================================
    while ~success

        attempt = attempt + 1;

        if attempt > max_attempt
            error('Model %d: exceed max attempts (%d)', mid, max_attempt);
        end

        % ---------- 随机场生成 ----------
        noise = rand(N,N);
        noise_pad = padarray(noise,[pad pad],0,'both');

        hsize = 2*ceil(3*sigma)+1;
        G = fspecial('gaussian',hsize,sigma);

        blur_pad = imfilter(noise_pad,G,'conv','same');
        blur = blur_pad(pad+1:pad+N, pad+1:pad+N);

        BW = blur > threshold;
        BW = bwareaopen(BW, min_area_px);

        % ---------- 连通域分析 ----------
        CC = bwconncomp(BW);
        num_cc = CC.NumObjects;

        if num_cc == N_scatter
            success = true;
        end
    end

    fprintf('   success after %d attempts\n', attempt);

    %% =================================================
    %  提取边界
    %% =================================================
    B = bwboundaries(BW,'noholes');

    %% =================================================
    %  新建 COMSOL 模型（仅几何）
    %% =================================================
    ModelUtil.clear;

    model = ModelUtil.create('Model');
    model.modelPath(pwd);
    model.label(sprintf('Scatter_%d_%04d', N_scatter, mid));

    model.component.create('comp1', true);
    model.component('comp1').geom.create('geom1', 2);
    model.component('comp1').geom('geom1').lengthUnit('m');

    geom = model.component('comp1').geom('geom1');

    %% ---------- 背景正方形（独立几何） ----------
    rect = geom.feature.create('rect1','Rectangle');
    rect.set('size', [L_domain L_domain]);
    rect.set('pos',  [-L_domain/2 -L_domain/2]);

    %% ---------- 散射体 Polygon（独立几何） ----------
    for k = 1:length(B)

        poly = B{k};

        % 降采样，避免点过多
        poly = poly(1:3:end,:);

        % 像素坐标 → 以原点为中心的物理坐标
        x = (poly(:,2) - (N/2 + 0.5)) * dx;
        y = ((N/2 + 0.5) - poly(:,1)) * dx;

        % 闭合多边形
        if x(1) ~= x(end) || y(1) ~= y(end)
            x(end+1) = x(1);
            y(end+1) = y(1);
        end

        pname = sprintf('scatter%d', k);
        p = geom.feature.create(pname, 'Polygon');
        p.set('x', x');
        p.set('y', y');
    end

    %% ---------- 构建并保存 ----------
    geom.run;

    fname = sprintf('Scatter_%d_%04d.mph', N_scatter, mid);
    mphsave(model, fullfile(outdir, fname));
end

disp('✅ 所有 COMSOL 几何模型生成完成（散射体数严格受控）');
