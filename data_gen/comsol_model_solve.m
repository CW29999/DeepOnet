clear;
clc;
import com.comsol.model.*
import com.comsol.model.util.*

model_dir = 'C:\Users\Admin\Desktop\大论文\算子拟合数据驱动\random_scatter_gen\model_3_scat\models_Scatter_3_gen';
save_dir = 'C:\Users\Admin\Desktop\大论文\算子拟合数据驱动\random_scatter_gen\model_3_scat\model_solved';

files = dir(fullfile(model_dir,'*.mph'));

disp('=== Debug: files ===');
disp(files);
disp(['Number of mph files found: ', num2str(length(files))]);

for i = 1 :length(files)
    fprintf('====Solving %d / %d : %s ====\n',i ,length(files), files(i).name);
    
    model_path = fullfile(files(i).folder, files(i).name);

    model = mphload(model_path);

    model.study('std1').run;

    save_name = fullfile(save_dir, files(i).name);

    mphsave(model, save_name);

    ModelUtil.remove('model');
end
disp('=== All models solved successfully ===');