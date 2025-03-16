%%
clear
load('data_St_g1.mat','index')
tnum = 4000;
value = tnum*(1-0.3);
idx = find(index > value, 1); %第一个满足条件的索引

%% S_sat-gam and VarN_sat-gam
data_num = 33;

data_tep = zeros(5,data_num);
for jj1 = 1: data_num
    FileName = ['data_St_g',num2str(jj1)];
    load (FileName,'Se_all','VarSl_all','gam','L','Dis_str','phase');

    %纠缠
    Se_tep1 = real(Se_all(idx:end,:));
    data_tep(1,jj1) = mean(Se_tep1,'all');      %均值
    data_tep(2,jj1) = std(Se_tep1,0,'all');     %标准差
    
    %涨落
    VarSl_tep1 = real(VarSl_all(idx:end,:));
    data_tep(3,jj1) = mean(VarSl_tep1,'all');
    data_tep(4,jj1) = std(VarSl_tep1,0,'all');

    data_tep(5,jj1) = gam;
end
Se_sat_aver = data_tep(1,:);
Se_sat_std = data_tep(2,:);
VarSl_sat_aver = data_tep(3,:);
VarSl_sat_std = data_tep(4,:);
gam_all = data_tep(5,:);

% save('data_L.mat','Se_sat_aver','Se_sat_std',...
%      'VarSl_sat_aver','VarSl_sat_std','gam_all','L')

figure
errorbar(gam_all, Se_sat_aver, Se_sat_std','-ob','displayname','S')

hold on
errorbar(gam_all,VarSl_sat_aver,VarSl_sat_std,'-or','displayname','Var N')
legend
xlabel('$\gamma$','interpreter','latex')
ylabel('$\bar{S}_{\infty}$, Var N','interpreter','latex')
title(['L= ',num2str(L),',D= ',num2str(Dis_str),',phase= ',num2str(phase)])