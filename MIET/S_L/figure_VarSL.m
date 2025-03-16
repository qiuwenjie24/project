%%
clear
load('data_VarSL_g1.mat','index')
tnum = 4000;
value = tnum*(1-0.3);
idx = find(index > value, 1); %第一个满足条件的索引

%% figure S-L
figure
num_data = 10;

for jj4 = 1:num_data
    
    FileName = ['data_VarSL_g',num2str(jj4)];
    load(FileName)
    
    Se_L_mean = zeros(1,L_num);
    Se_L_std = zeros(1,L_num);
    VarS_L_mean = zeros(1,L_num);
    VarS_L_std = zeros(1,L_num);
    
    for jj3=1:L_num
        %纠缠
        tep1 = Se_L_all{jj3};
        tep1 = real(tep1(idx:end,:));
        Se_L_mean(jj3)=mean(tep1,'all');
        Se_L_std(jj3)=std(tep1,0,'all');
        
        %涨落
        tep2 = VarS_L_all{jj3};
        tep2 = real(tep2(idx:end,:));
        VarS_L_mean(jj3)=mean(tep2,'all');
        VarS_L_std(jj3)=std(tep2,0,'all');
    end
    
    leg_str = ['gam=',num2str(gam)];
    % errorbar(L_all,Se_L_mean,Se_L_std,'-o','displayname',leg_str)
    hold on
    errorbar(L_all,VarS_L_mean,VarS_L_std,'-o','displayname',leg_str)
end

legend
xlabel('L')
ylabel('Var$\bar{S} (L/2,L)$','interpreter','latex')
set(gca,'XScale','log')
title(['D=',num2str(Dis_str),', phase=',num2str(phase)],'FontSize',14)
axis([-inf inf 0 inf])