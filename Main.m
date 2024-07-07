clear;
close all;
clc;
addpath('.\help_functions\');

%% Import parameters
disp('import parameters');
load("parameters_Actin.mat")
stackfilename = 'FO-3DSIM.tif';

tic

%% Read data
disp('read data');
img = imstackread('Actin_488nm_1.49NA_65nm.tif');
[Nx,Ny,Nz] = size(img);
Nz = Nz/15;
raw_img = zeros(Nx,Ny,Nz,3,5);
for jangle = 1:3
    for jstep = 1:5
        for jz = 1:Nz
            raw_img(:,:,jz,jangle,jstep) = img(:,:,15*(jz-1)+5*(jangle-1)+jstep);
            %raw_img(:,:,jz,jangle,jstep) = img(:,:,5*Nz*(jangle-1)+5*(jz-1)+jstep);
        end
    end
end
raw_img_gpu = gpuArray(single(raw_img));
raw_img = gpuArray(single(imresize(raw_img,2)));

%% WF
disp('generate WF');
WF = gpuArray(single(zeros(Nx,Ny,Nz)));
for jangle = 1:3
    for jstep = 1:5
        WF = sum(sum(raw_img_gpu,4),5)/3;
    end
end

%% HiLo
disp('generate HiLo');
HiLo_img = zeros(Nx,Ny,Nz);
for jz = 1:Nz
    temp_img = squeeze((raw_img_gpu(:,:,jz,1,1)+raw_img_gpu(:,:,jz,2,1)+raw_img_gpu(:,:,jz,3,1)))/3;
    [HiLo_img(:,:,jz)] = HiLo1(WF(:,:,jz),temp_img,NA,emwavelength,pixelsize(1));
end
HiLo_img = gather(max(WF(:))*HiLo_img./max(HiLo_img(:)));

%% Seperate matrix
disp('generate matrix');
A = gpuArray(single([1          1           1           1           1;
    cos(0) cos(2*pi/5) cos(4*pi/5) cos(6*pi/5) cos(8*pi/5);
    sin(0) sin(2*pi/5) sin(4*pi/5) sin(6*pi/5) sin(8*pi/5);
    cos(0) cos(4*pi/5) cos(8*pi/5) cos(12*pi/5) cos(16*pi/5);
    sin(0) sin(4*pi/5) sin(8*pi/5) sin(12*pi/5) sin(16*pi/5)]));
spatial_matrix = inv(A);

%% Coefficient
coefficientFuncs = gpuArray(single(zeros(2*Nx,2*Ny,5,3)));
for jangle = 1:3
    coefficientFuncs(:,:,:,jangle) = Get_coefficient(raw_img,spatial_matrix,freq,pha,jangle,allmodule);
end

%% Generate initial
disp('generate initial');
SIM_image_initial = zeros(2*Nx,2*Ny,Nz);
for jz = 1:Nz
    for jangle = 1:3
        for jstep = 1:5
            SIM_image_initial(:,:,jz) = SIM_image_initial(:,:,jz) + raw_img(:,:,jz,jangle,jstep).*coefficientFuncs(:,:,jstep,jangle);
        end
    end
end

%% Notch filter
disp('notch filter');
SIM_freq_notch = fftshift(fftn(SIM_image_initial));
max_SIM_freq_notch = abs(max(SIM_freq_notch(:)));
for jj = 1:15
    if  jj ~=1 &&jj ~=6 && jj ~= 11
        SIM_freq_notch = SIM_freq_notch .* notch(:,:,:,jj);
    end
end
SIM_freq_notch = max_SIM_freq_notch.*SIM_freq_notch./max(abs(SIM_freq_notch(:)));
SIM_freq_notch = SIM_freq_notch + fftshift(fftn(spatial_matrix(1,1).*imresize(HiLo_img,2)))/3;

%% Two Filters
disp('two filters');
SIM_image_notched = abs(ifftn(ifftshift(SIM_freq_notch.*Filter1.*Filter2)));

toc

%% FO-3DSIM result
final_image_notch = SIM_image_notched;
final_image_notch = uint16(65535*final_image_notch./max(final_image_notch(:)));
for jz = 1:Nz
    imwrite(final_image_notch(:,:,jz), stackfilename, 'WriteMode','append') % 写入stack图像
end


