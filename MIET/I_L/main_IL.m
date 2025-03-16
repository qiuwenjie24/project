%% AAH I-L
clear

%% parameter
Dis_str = 0.0; %disorder strength
phase = 0;   %disorder phase
aver_num = 100;     %average times of random 100
tnum = 6000;        %time number  6000,T=300

g_all = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 6];
g_num = max(size(g_all)); %10

L_all = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, ... 
        160, 240, 320, 400];
L_num = max(size(L_all)); %15

%%
for jj2 = 1:g_num
    
gam = g_all(jj2);   %measurement strength
    
datetime
tic
Ig_L_all = cell(1,L_num);
VarIg_L_all = cell(1,L_num);
  for jj1 = 1:L_num

    L = L_all(jj1);   %total length
    [Ig_all, VarIg_all, index] = fn_AAH_It(L, gam, Dis_str, aver_num, tnum, phase);

    Ig_L_all{jj1} = Ig_all;
    VarIg_L_all{jj1} = VarIg_all;

    fprintf('%d - %d end \n',jj2,jj1)
  end
endtime=toc()
datetime

FileName = ['data_VarIL_g',num2str(jj2)];
save(FileName,'gam','L_all','Dis_str','L_num',...
     'Ig_L_all','VarIg_L_all','endtime','phase','index')

end
