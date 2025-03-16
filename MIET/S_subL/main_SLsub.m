%%
clear

%% parameter
Dis_str = 0.0; %disorder strength
L = 610;
aver_num = 50;
tnum = 4000;
phase=0;

g_all = linspace(0.05,1,20);
g_num = max(size(g_all)); %20

temp1 = exp( [linspace(0,log(L/2)*8/9,18), linspace(log(L/2)*8/9,log(L/2),12)] );
temp2 = unique( round(temp1) );
Lsub_all = temp2;
l_num = max(size(Lsub_all)); %27

%%
for jj2 = 1:g_num
jj2
gam = g_all(jj2);   %measurement strength

datetime
tic

[Se_sub_all, VarSl_sub_all, index] = fn_AAH_SLsub(L, gam, Dis_str, aver_num, tnum, phase, Lsub_all);

endtime=toc()
datetime

FileName = ['data_VarSLsub_g',num2str(jj2)];
save(FileName,'gam','Lsub_all','Dis_str','l_num','L',...
    'Se_sub_all','VarSl_sub_all','endtime','phase','index')
end



