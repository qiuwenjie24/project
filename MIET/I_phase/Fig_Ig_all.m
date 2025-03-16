%% Ig_sat-gam
%%
clear
load('data_VarI_g1.mat','index')
tnum = 6000;
value = tnum*(1-0.3);
idx = find(index > value, 1); %第一个满足条件的索引

%%
data_num = 33;
aver = 200;
data_all = zeros(7,data_num);

for jj1 = 1: data_num
    FileName = ['data_VarI_g',num2str(jj1)];
    load (FileName,'Ig_all','VarIg_all','gam','L','Dis_str','phase');
    
    %互信息
    Ig_temp = real(Ig_all(index:end,:));
    data_all(1,jj1) = mean(Ig_temp,'all');      %均值
    data_all(2,jj1) = std(Ig_temp,0,'all');     %标准差
    temp1 = mean(Ig_temp,2);        %Y = (X1 + X2 + ... + Xn)/n
    data_all(3,jj1) = std(temp1,0,'all');      %Y的标准差
    
    %互涨落
    VarIg_temp = real(VarIg_all(index:end,:));
    data_all(4,jj1) = mean(VarIg_temp,'all');
    data_all(5,jj1) = std(VarIg_temp,0,'all');
    temp2 = mean(VarIg_temp,2);        %Y = (X1 + X2 + ... + Xn)/n
    data_all(6,jj1) = std(temp2,0,'all');      %Y的标准差
    
    data_all(7,jj1) = gam;
end

Ig_sat_aver = data_all(1,:);
Ig_sat_std = data_all(2,:);
Ig_sat_std_new = data_all(3,:);
VarIg_sat_aver = data_all(4,:);
VarIg_sat_std = data_all(5,:);
VarIg_sat_std_new = data_all(6,:);
gam_all = data_all(7,:);

% save('data_L.mat','Ig_sat_aver','Ig_sat_std_new','Ig_sat_std_new',...
%      'VarIg_sat_aver','VarIg_sat_std','VarIg_sat_std_new','gam_all','L')

figure
errorbar(gam_all, Ig_sat_aver, Ig_sat_std','-ob','displayname','I')

hold on
errorbar(gam_all, VarIg_sat_aver, VarIg_sat_std,'-or','displayname','Var I')
legend
xlabel('$\gamma$','interpreter','latex')
ylabel('$\bar{I}_{g}$, Var Ig','interpreter','latex')
title(['L= ',num2str(L),',D= ',num2str(Dis_str),',phase= ',num2str(phase)])