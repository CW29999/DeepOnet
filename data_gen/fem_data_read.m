clc; clear


model = mphload("model_solved\Scatter_3_0001.mph");
solinfo = mphsolinfo(model,"soltag",'sol1');


dset = solinfo.dataset;
num_samples = 64*64;

x = linspace(-5e-6, 5e-6, sqrt(num_samples));
y = linspace(-5e-6, 5e-6, sqrt(num_samples));
[X, Y] = meshgrid(x, y);
xx = [X(:)'; Y(:)'];



num_sensors = max(size(xx));


Eplison = mphinterp(model,'ewfd.epsilonrxx','coord', xx,dataset='dset1',outersolnum='all');
Ez = mphinterp(model,'ewfd.Ez','coord',xx,'dataset','dset1',outersolnum='all');
sizeof_eps = size(Eplison);
num_dataset = sizeof_eps(1);
net_y = xx';
net_x = xx';
net_u = reshape(Eplison, [num_samples,num_dataset]).';
net_Gu_re = reshape(Ez,[num_samples,num_dataset]).';

net_Gu_re_real = real(net_Gu_re);  % 实部
net_Gu_re_imag = imag(net_Gu_re);  % 虚部

% 沿着第三维拼接
net_Gu_re = cat(3, net_Gu_re_real, net_Gu_re_imag);

%随机划分数据集
pctTrain = 0.8;
idx = randperm(num_dataset);
nTrain = floor(pctTrain * num_dataset);
trainIdx = idx(1:nTrain);
testIdx = idx(nTrain+1:end);


X_train = net_y;
X_test  = net_y;

Eplison_train = net_u(trainIdx, :);
Eplison_test = net_u(testIdx, :);

Ez_train = net_Gu_re(trainIdx, :,:);

Ez_test = net_Gu_re(testIdx, :,:);



save("deepOnet_scat_data_4096.mat","Eplison_test",'Eplison_train','X_test','X_train','Ez_train','Ez_test')
