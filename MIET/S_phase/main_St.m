%% AAH S-t
clear 

%% parameter
Dis_str = 0.0; %disorder strength
L = 100;        %total length 
phase = 0;   %disorder phase
aver_num = 100;     %average times of random 100
tnum = 4000;        %time number  4000,T=200

g_all = (0.06:0.02:0.7);
g_num = max(size(g_all));  %33

%%
for jj1 = 1:g_num
jj1
datetime
tic

gam = g_all(jj1);   %measurement strength
[Se_all, VarSl_all, index] = fn_AAH_St(L, gam, Dis_str, aver_num, tnum, phase);

endtime=toc()
datetime

FileName = ['data_St_g',num2str(jj1)];
save(FileName,'gam','L','Dis_str','Se_all','VarSl_all','endtime','phase','index')

end
