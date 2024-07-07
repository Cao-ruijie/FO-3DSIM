function coefficientFuncs = Get_coefficient(Img,matrix,freq,pha,jangle,allmodule)
[Nx,Ny,~] = size(Img);
SpaFreqX = freq(jangle,1);
SpaFreqY = freq(jangle,2);
u0 = gpuArray(single((2*pi*SpaFreqX)/Nx));
v0 = gpuArray(single((2*pi*SpaFreqY)/Ny));
r = gpuArray(single(0:Nx-1));
c = gpuArray(single(0:Ny-1));
[R, C] = meshgrid(r, c);

allmodule1 = allmodule(1);
allmodule2 = allmodule(2);

if allmodule1<0.3
    allmodule1=0.3;
end
if allmodule2<0.2
    allmodule2=0.2;
end
first_order = cos(u0*R+v0*C+pha(jangle))/allmodule1;
first_order2 = sin(u0*R+v0*C+pha(jangle))/allmodule1;
second_order = cos(2*(u0*R+v0*C+pha(jangle)))/allmodule2;
second_order2 = sin(2*(u0*R+v0*C+pha(jangle)))/allmodule2;

coefficientFuncs(:,:,1) =  matrix(1,2).*first_order  +  matrix(1,3).*first_order2...
                                      + matrix(1,4).*second_order + matrix(1,5).*second_order2;
coefficientFuncs(:,:,2) =  matrix(2,2).*first_order  +  matrix(2,3).*first_order2...
                                      + matrix(2,4).*second_order + matrix(2,5).*second_order2;
coefficientFuncs(:,:,3) =  matrix(3,2).*first_order  +  matrix(3,3).*first_order2...
                                      + matrix(3,4).*second_order + matrix(3,5).*second_order2;
coefficientFuncs(:,:,4) =  matrix(4,2).*first_order  +  matrix(4,3).*first_order2...
                                      + matrix(4,4).*second_order + matrix(4,5).*second_order2;
coefficientFuncs(:,:,5) =  matrix(5,2).*first_order  +  matrix(5,3).*first_order2...
                                      + matrix(5,4).*second_order + matrix(5,5).*second_order2;
% coefficientFuncs(:,:,1) = matrix(1,1) + matrix(1,2).*first_order  +  matrix(1,3).*first_order2...
%                                       + matrix(1,4).*second_order + matrix(1,5).*second_order2;
% coefficientFuncs(:,:,2) = matrix(2,1) + matrix(2,2).*first_order  +  matrix(2,3).*first_order2...
%                                       + matrix(2,4).*second_order + matrix(2,5).*second_order2;
% coefficientFuncs(:,:,3) = matrix(3,1) + matrix(3,2).*first_order  +  matrix(3,3).*first_order2...
%                                       + matrix(3,4).*second_order + matrix(3,5).*second_order2;
% coefficientFuncs(:,:,4) = matrix(4,1) + matrix(4,2).*first_order  +  matrix(4,3).*first_order2...
%                                       + matrix(4,4).*second_order + matrix(4,5).*second_order2;
% coefficientFuncs(:,:,5) = matrix(5,1) + matrix(5,2).*first_order  +  matrix(5,3).*first_order2...
%                                       + matrix(5,4).*second_order + matrix(5,5).*second_order2;
end