%% AAH S-L
clear

%% parameter
Dis_str = 0.0; %disorder strength
phase = 0;   %disorder phase
aver_num = 100;     %average times of random 100
tnum = 4000;        %time number  4000,T=200


g_all = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 6];
g_num = max(size(g_all)); %10

L_all = [10, 20, 30, 40, 50, 60, 70, 80, 90, ...
         100, 150, 200, 250, 300, 400];
L_num = max(size(L_all)); %15

%%
for jj2 = 1:g_num
    
gam = g_all(jj2);   %measurement strength
    
datetime
tic
Se_L_all = cell(1,L_num);
VarS_L_all = cell(1,L_num);
  for jj1 = 1:L_num

    L = L_all(jj1);   %total length
    [Se_all, VarSl_all, index] = fn_AAH_St(L, gam, Dis_str, aver_num, tnum, phase);

    Se_L_all{jj1} = Se_all;
    VarS_L_all{jj1} = VarSl_all;

    fprintf('%d - %d end \n',jj2,jj1)
  end
endtime=toc()
datetime

FileName = ['data_VarSL_g',num2str(jj2)];
save(FileName,'gam','L_all','Dis_str','L_num',...
     'Se_L_all','VarS_L_all','endtime','phase','index')

end

