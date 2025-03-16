%%
clear

%% parameter
Dis_str_all = [1.5, 2.0, 2.5]; %disorder strength
L = 400;
aver_num = 1;
tnum = 4000;
phase=0;
gam = 0;   %measurement strength


temp1 = exp( linspace(0,log(L/2),30) );
temp2 = unique( round(temp1) );
Lsub_all = temp2;
l_num = max(size(Lsub_all)); %26

%%
datetime

for jj2 = 1:3
tic
Dis_str = Dis_str_all(jj2);
[Se_sub_all, VarSl_sub_all, index] = fn_AAH_SLsub_test(L, gam, Dis_str, aver_num, tnum, phase, Lsub_all);

endtime=toc()
datetime

FileName = ['data_VarSLsub_g',num2str(jj2)];
save(FileName,'gam','Lsub_all','Dis_str','l_num','L',...
    'Se_sub_all','VarSl_sub_all','endtime','phase','index')

end


