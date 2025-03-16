%%
% clear
% load('data_VarSLsub_g1.mat')
% figure
% plot(0.05*index, real(Se_sub_all(:,1,9)) )
%%
clear
load('data_VarSLsub_g1.mat','index')
tnum = 4000;
value = tnum*(1-0.3);
idx = find(index > value, 1); %第一个满足条件的索引

%% S-logl
figure
g_num = 20;
for jj1 = 1:g_num
    
    FileName = ['data_VarSLsub_g',num2str(jj1)];
    load(FileName)
    
    %纠缠
    S_sub_mean = zeros(max(size(Lsub_all)),1);
    S_temp1 = mean(Se_sub_all,2);  %平均路径
    S_temp2 = mean(S_temp1(idx:end,1,:),1);  %平均饱和时间
    S_sub_mean(:) = S_temp2(1,1,:);
    
    %涨落
    VarS_sub_mean = zeros(max(size(Lsub_all)),1);
    VarS_temp1 = mean(VarSl_sub_all,2);
    VarS_temp2 = mean(VarS_temp1(idx:end,1,:),1); 
    VarS_sub_mean(:) = VarS_temp2(1,1,:);

%     X = log2( sin( pi.*Lsub_all/L ) *L/pi ) /3;
    X = sin( pi.*Lsub_all/L );
    Y = real(S_sub_mean );
    leg_str = ['$\gamma$=',num2str(gam)];
    hold on
    h1 = plot(X, Y, 'o-','DisplayName',leg_str);
    set(h1, 'markerfacecolor', get(h1, 'color'));

end
legend
legend('Location','northwest','Interpreter','latex','FontSize',14)
xlabel('$\sin(\pi l/L)$','interpreter','latex','FontSize',14)
ylabel('$\bar{S}_{\Omega}(l,L=610)$','interpreter','latex','FontSize',14)
set(gca,'XScale','log')
title(['Dis_{str}=',num2str(Dis_str),', phase=',num2str(phase)],'FontSize',14)
box on

