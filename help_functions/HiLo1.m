function [HiLo_final] = HiLo1(WF,raw_img,NA,lambta,pixelsize)

%% iter
uniform_file = WF;

%% Define parameters
% res = Lateral_resolution*1e6*0.0254;    % Lateral resolution
[h, w] = size(uniform_file);
res = 0.61 * lambta / (NA);                   % resolution
k_m = w / (res / pixelsize);           % objective cut-off frequency ???
kc = nearest(k_m * 0.2);                % cut-off frequency between hp and lp filter
k_side_HP = kc/2;                     % cutoff frequency of high-pass filter for differential image
sigmaLP = kc*2/2.355;                   % Finding sigma value for low pass
lambda = nearest(w/(2*kc));             % Side length of contrast
% evalutation mask by rounding to
% nearest odd integer
if mod(lambda,2) == 0                   % lambda must be odd
    lambda = lambda+1;
else
end

gauss_corr = 1;

h = h+2*lambda;                         % increase image size by lambda for
w = w+2*lambda;                         % padding

%% Creating filters, pre-allocating and reading image stacks

% Create band pass, high and low pass filters
lp = lpgauss(h,w,sigmaLP);
hp = hpgauss(h,w,sigmaLP);
hp_1 = hpgauss(h,w,sigmaLP * 2);
hp_side = hp_OneSide(h,w,k_side_HP);
u = single(uniform_file);
uni = padarray(u./gauss_corr,[lambda lambda],'symmetric');
s = single(raw_img);
rat = padarray(s./gauss_corr,[lambda lambda],'symmetric');
rat = rat./(uni+sqrt (eps));


%% One-side highpass filtering ratio image stack and Contrast evaluation
rat_hp_s_f = fft2(rat).*hp_side;
rat_hp_s_f = rat_hp_s_f .* hp_1;
rat_hp_s = ifft2(rat_hp_s_f);
weight = sqrt(rat_hp_s .* conj(rat_hp_s));
intermediate_weight = weight(lambda+1:end-lambda,lambda+1:end-lambda,:);
if max(intermediate_weight(:))~=0
    weight = weight/max(intermediate_weight(:));
end
Lo = weight .* uni;

%% Filtering
Hi = real(ifft2(fft2(uni).*hp));
Hi = Hi(lambda+1:end-lambda,lambda+1:end-lambda,:);
Lo = real(ifft2(fft2(Lo).*lp));
Lo = Lo(lambda+1:end-lambda,lambda+1:end-lambda,:);
Hif=fft2(Hi);Lof=fft2(Lo);
nabla =abs((sum(sum(Hif)))./(sum(sum(Lof))));

%% Scaling
Lo = Lo*nabla;

%% Reconstruction and 16bit conversion
HiLo_final = Hi + Lo;
end



function [ out ] = hpgauss(H,W,SIGMA)
%   Creates a 2D Gaussian filter for a Fourier space image of height H and
%   width W. SIGMA is the standard deviation of the Gaussian.
out=1-lpgauss(H,W,SIGMA);
end

function [ out ] = lpgauss(H,W,SIGMA)
%   Creates a 2D Gaussian filter for a Fourier space image
%   W is the number of columns of the source image and H is the number of
%   rows. SIGMA is the standard deviation of the Gaussian.
H = single(H);
W = single(W);
kcx = (SIGMA);
kcy = ((H/W)*SIGMA);
[x,y] = meshgrid(-floor(W/2):floor((W-1)/2), -floor(H/2):floor((H-1)/2));
temp = -gpuArray(single((x.^2/(kcx^2)+y.^2/(kcy^2))));
out = ifftshift(exp(temp));
% out = ifftshift(exp(-(x.^2/(kcx^2)+y.^2/(kcy^2))));
end

function [ out ] = hp_OneSide(H,W,SIGMA)
%   Creates a 2D Gaussian filter for a Fourier space image of height H and
%   width W. SIGMA is the standard deviation of the Gaussian.

H = single(H);
W = single(W);
kcx = (SIGMA);
kcy = ((H/W)*SIGMA);
[x,y] = meshgrid(-floor(W/2):floor((W-1)/2), -floor(H/2):floor((H-1)/2));
lp_out = gpuArray(single(exp(-(x.^2/(kcx^2)+y.^2/(kcy^2)))));

out=1-lp_out;

H_c = floor(H/2);
W_c = floor(W/2);

out(:, 1:W_c) = 0;
out(1: (H_c+1), W_c+1) = 0;
out = ifftshift(out);
end
