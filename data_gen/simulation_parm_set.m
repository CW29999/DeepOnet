clc;clear;
import com.comsol.model.*
import com.comsol.model.util.*

model_dir = 'models_Scatter_3_gen';
files = dir(fullfile(model_dir,'Scatter_3_*.mph'));

for i = 1:length(files)
    fprintf('Processing %s (%d/%d)\n', ...
        files(i).name, i, length(files));

    model_path = fullfile(model_dir, files(i).name);
    model = mphload(model_path);

    geom = model.component('comp1').geom('geom1');
    geom.run;
    
    info = mphgeominfo(model,'geom1');
    last_bundary = info.Nboundaries;
    
    
    model.param.set('epsr1', '2');
    model.param.descr('epsr1', '');
    model.param.set('epsr2', '2');
    model.param.descr('epsr2', '');
    model.param.set('epsr3', '2');
    model.param.descr('epsr3', '');
    model.param.set('epsr_air', '1');
    model.param.descr('epsr_air', '');
    model.param.set('lda', '2.1[um]');
    model.param.descr('lda', '');
    model.param.set('Einc_z', '1');
    model.param.descr('Einc_z', '');
    model.param.set('k0', '2*pi/lda');
    model.param.descr('k0', '');
    model.param.set('theta', '0');
    model.param.descr('theta', '');
    model.param.set('kx', 'k0*cos(theta)');
    model.param.descr('kx', '');
    model.param.set('ky', 'k0*sin(theta)');
    model.param.descr('ky', '');
    
    model.component('comp1').material.create('mat1', 'Common');
    model.component('comp1').material.create('mat2', 'Common');
    model.component('comp1').material.create('mat3', 'Common');
    model.component('comp1').material.create('mat4', 'Common');
    model.component('comp1').material('mat1').label('air');
    model.component('comp1').material('mat1').propertyGroup('def').set('relpermittivity', '');
    model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', '');
    model.component('comp1').material('mat1').propertyGroup('def').set('relpermeability', '');
    model.component('comp1').material('mat1').propertyGroup('def').set('relpermittivity', {'epsr_air'});
    model.component('comp1').material('mat1').propertyGroup('def').set('electricconductivity', {'0'});
    model.component('comp1').material('mat1').propertyGroup('def').set('relpermeability', {'1'});
    model.component('comp1').material('mat2').label('Scatter1');
    model.component('comp1').material('mat3').label('Scatter2');
    model.component('comp1').material('mat4').label('Scatter4');
    model.component('comp1').material('mat2').propertyGroup('def').set('relpermittivity', '');
    model.component('comp1').material('mat2').propertyGroup('def').set('relpermeability', '');
    model.component('comp1').material('mat2').propertyGroup('def').set('electricconductivity', '');
    model.component('comp1').material('mat2').propertyGroup('def').set('relpermittivity', {'epsr1'});
    model.component('comp1').material('mat2').propertyGroup('def').set('relpermeability', {'1'});
    model.component('comp1').material('mat2').propertyGroup('def').set('electricconductivity', {'0'});
    model.component('comp1').material('mat2').selection.set([2]);
    model.component('comp1').material('mat3').propertyGroup('def').set('relpermittivity', '');
    model.component('comp1').material('mat3').propertyGroup('def').set('electricconductivity', '');
    model.component('comp1').material('mat3').propertyGroup('def').set('relpermeability', '');
    model.component('comp1').material('mat3').propertyGroup('def').set('relpermittivity', {'epsr2'});
    model.component('comp1').material('mat3').propertyGroup('def').set('electricconductivity', {'0'});
    model.component('comp1').material('mat3').propertyGroup('def').set('relpermeability', {'1'});
    model.component('comp1').material('mat3').selection.set([3]);
    model.component('comp1').material('mat4').propertyGroup('def').set('relpermittivity', '');
    model.component('comp1').material('mat4').propertyGroup('def').set('electricconductivity', '');
    model.component('comp1').material('mat4').propertyGroup('def').set('relpermeability', '');
    model.component('comp1').material('mat4').propertyGroup('def').set('relpermittivity', {'epsr3'});
    model.component('comp1').material('mat4').propertyGroup('def').set('electricconductivity', {'0'});
    model.component('comp1').material('mat4').propertyGroup('def').set('relpermeability', {'1'});
    model.component('comp1').material('mat4').selection.set([4]);
    
    model.component('comp1').physics.create('ewfd', 'ElectromagneticWavesFrequencyDomain', 'geom1');
    model.component('comp1').physics('ewfd').create('sctr1', 'Scattering', 1);
    model.component('comp1').physics('ewfd').feature('sctr1').selection.set([1 2 3 last_bundary]);
    model.component('comp1').physics('ewfd').prop('BackgroundField').set('SolveFor', 'scatteredField');
    model.component('comp1').physics('ewfd').prop('BackgroundField').set('Eb', {'0' '0' 'Einc_z*exp(-1i*(kx*x+ky*y))'});
    model.component('comp1').physics('ewfd').feature('wee1').set('DisplacementFieldModel', 'RelativePermittivity');
    
    
    model.component('comp1').mesh.create('mesh1');
    model.component('comp1').mesh('mesh1').create('size1', 'Size');
    model.component('comp1').mesh('mesh1').run;
    model.component('comp1').mesh('mesh1').create('ftri1', 'FreeTri');
    model.component('comp1').mesh('mesh1').run;
    
    model.study.create('std1');
    model.study('std1').create('freq', 'Frequency');
    model.study('std1').feature('freq').set('solnum', 'auto');
    model.study('std1').feature('freq').set('notsolnum', 'auto');
    model.study('std1').feature('freq').set('ngen', '1');
    model.study('std1').feature('freq').set('ngenactive', false);
    model.study('std1').feature('freq').setSolveFor('/physics/ewfd', true);
    model.study('std1').feature('freq').set('plist', 'c_const/lda');
    model.study('std1').create('param', 'Parametric');
    model.study('std1').feature('param').set('sweeptype', 'filled');
    model.study('std1').feature('param').setIndex('pname', 'epsr1', 0);
    model.study('std1').feature('param').setIndex('plistarr', '', 0);
    model.study('std1').feature('param').setIndex('punit', '', 0);
    model.study('std1').feature('param').setIndex('pname', 'epsr1', 0);
    model.study('std1').feature('param').setIndex('plistarr', '', 0);
    model.study('std1').feature('param').setIndex('punit', '', 0);
    model.study('std1').feature('param').setIndex('plistarr', 'range(1,0.1,2)', 0);
    model.study('std1').feature('param').setIndex('pname', 'epsr2', 1);
    model.study('std1').feature('param').setIndex('plistarr', '', 1);
    model.study('std1').feature('param').setIndex('punit', '', 1);
    model.study('std1').feature('param').setIndex('pname', 'epsr2', 1);
    model.study('std1').feature('param').setIndex('plistarr', '', 1);
    model.study('std1').feature('param').setIndex('punit', '', 1);
    model.study('std1').feature('param').setIndex('plistarr', 'range(2,0.1,3)', 1);
    model.study('std1').feature('param').setIndex('pname', 'epsr3', 2);
    model.study('std1').feature('param').setIndex('plistarr', '', 2);
    model.study('std1').feature('param').setIndex('punit', '', 2);
    model.study('std1').feature('param').setIndex('pname', 'epsr3', 2);
    model.study('std1').feature('param').setIndex('plistarr', '', 2);
    model.study('std1').feature('param').setIndex('punit', '', 2);
    model.study('std1').feature('param').setIndex('plistarr', 'range(1.5,0.1,2.5)', 2);
    
    mphsave(model,model_path);
    ModelUtil.remove('Model');

end

disp('All models processed');