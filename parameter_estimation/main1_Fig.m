%% figure-S1
close all
clear
FileName = ['estimate1'];
load(FileName)
%% figure Shanon Entropy
figure

%Num_aver*Tnum  to  1*Tnum
vE_sh_aver = mean(vE_sh_all, 1);
% vE_sh_std  = std(vE_sh_all,0,1);

plot((0:Tnum-1),vE_sh_aver, '-o');
xlabel('$t/\beta$','interpret','latex');
ylabel('Shanon Entropy');
title(['S--$\beta$=',num2str(beta),', $\lambda=$',num2str(lambda)],...
       'interpret','latex')

%% figure Fisher information
figure

%Num_aver*tnum  to  1*tnum
vF_aver = mean(vF_all, 1); 
% vF_std = std(vF_all,0,1);

plot( (1:Tnum),vF_aver, '-o');
xlabel('$t/\beta$','interpret','latex');
ylabel('Fisher information');
title(['S--$\beta$=',num2str(beta),', $\lambda=$',num2str(lambda)],...
       'interpret','latex')

%% figure Von Neumann Entropy
figure

%Num_aver*Tnum  to  1*Tnum
vE_vn_all = real(vE_vn_all);
vE_vn_aver = mean(vE_vn_all, 1);
% vE_vn_std  = std(vE_vn_all,0,1);

plot((0:Tnum-1),vE_vn_aver, '-o');
xlabel('$t/\beta$','interpret','latex');
ylabel('von Neumann Entropy');
title(['S--$\beta$=',num2str(beta),', $\lambda=$',num2str(lambda)],...
       'interpret','latex')

