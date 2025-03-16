clear
pos1=[0.13,0.25,0.35 0.57]; pos2=[0.57,0.23,0.35 0.57];
figure

%% 
subplot('position',pos1);
load('.\no2\c5\mic_OTOC_all_no2_c5.mat')  

time=0:0.001:3;
plot(time,c_n_350,'Color',[0.4940 0.1840 0.5560],'LineWidth',2)
hold on
plot(time,c_n_400,'Color',[0.9290 0.6940 0.1250],'LineWidth',2)
hold on
plot(time,c_n_450,'Color',[0.8500 0.3250 0.0980],'LineWidth',2)
hold on
plot(time,c_n_500,'Color',[0 0.4470 0.7410],'LineWidth',2)

set(gca,'FontSize',30,'linewidth',3)  %设置坐标轴字体大小,线粗
text(0.1,630,'(a) $\eta=\frac{1}{3},c=10^{12}$','interpreter','latex','fontsize',35)
xlabel('t','fontsize',35)
ylabel('$c_{200}(t)$','interpreter','latex','fontsize',35);
legend('$N_{trunc}=350$','$N_{trunc}=400$','$N_{trunc}=450$','$N_{trunc}=500$',...
    'interpreter','latex','fontsize',17);
axis([0,3,0,700])


%%
subplot('position',pos2);
load('.\no2\c5\ther_OTOC_all_no2_c5.mat')

time=0:0.001:3;
plot(time,C_T_1,'Color',[0.4940 0.1840 0.5560],'LineWidth',3)
hold on
plot(time,C_T_2,'Color',[0.9290 0.6940 0.1250],'LineWidth',3) 
hold on
plot(time,C_T_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',3)
hold on
plot(time,C_T_4,'Color',[0 0.4470 0.7410],'LineWidth',3)

set(gca,'FontSize',35,'linewidth',3)  %设置坐标轴字体大小,线粗
text(0.1,65,'(b) $\eta=\frac{1}{3},c=10^{12}$','interpreter','latex','fontsize',35)
xlabel('t','fontsize',30),
ylabel('C_T(t)','fontsize',28);
legend('T=100','T=200','T=300','T=400','fontsize',17);
axis([0,3,0,70])
