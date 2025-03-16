%%
clear
load('data_VarIL_g1.mat','index')
tnum = 6000;
value = tnum*(1-0.3);
idx = find(index > value, 1); %第一个满足条件的索引

%% VarI
figure
num_g = 10;
L_num = 15;
aver_num = 100;

for jj1 = 1:num_g
    %%
    Ig_L_mean = zeros(1,L_num);
    Ig_L_std = zeros(1,L_num);
    VarIg_L_mean = zeros(1,L_num);
    VarIg_L_std = zeros(1,L_num);
    
    FileName = ['data_VarIL_g',num2str(jj1)];
    load(FileName)
    
    for jj3 = 1:L_num
        %互信息
        I_temp1 = Ig_L_all{jj3};
        I_temp2 = real(I_temp1(idx:end,:));
        Ig_L_mean(jj3) = mean(I_temp2,'all');
        Ig_L_std(jj3) = std(I_temp2,0,'all');
        %互涨落
        VarI_temp1 = VarIg_L_all{jj3};
        VarI_temp2 = real(VarI_temp1(idx:end,:));
        VarIg_L_mean(jj3) = mean(VarI_temp2,'all');
        VarIg_L_std(jj3) = std(VarI_temp2,0,'all');
    end
    
    %%
    leg_str = ['$\gamma$=',num2str(gam)];
    hold on 
    h1 = plot(L_all, VarIg_L_mean, 'o-','DisplayName',leg_str);
    set(h1, 'markerfacecolor', get(h1, 'color'));
    
end

legend
legend('Location','northwest','Interpreter','latex','FontSize',10)
xlabel('L','FontSize',14)
ylabel('$\bar{VarI}_{AB}(r_{AB}=L/2)$','interpreter','latex','FontSize',14)
set(gca,'XScale','log')
box on
title(['D=',num2str(Dis_str),', phase=',num2str(phase)],'FontSize',14)
axis([-inf inf 0 inf])