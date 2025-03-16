%% AAH I-t
clear

%% parameter
Dis_str = 0.0; %disorder strength
L = 80;     %total length 8的倍数 160 240 320 400
phase = 0;   %disorder phase
aver_num = 200;     %average times of random 200
tnum = 6000;        %time number  6000,T=300

g_all = (0.06:0.02:0.7);
g_num = max(size(g_all));  %33

%%
for jj1 = 1:g_num
jj1
datetime
tic

gam = g_all(jj1);   %measurement strength
[Ig_all, VarIg_all, index] = fn_AAH_It(L, gam, Dis_str, aver_num, tnum, phase);

endtime=toc()
datetime

FileName = ['data_VarI_g',num2str(jj1)];
save(FileName,'gam','L','Dis_str',...
     'Ig_all','VarIg_all','endtime','phase','g_all','index')
end


