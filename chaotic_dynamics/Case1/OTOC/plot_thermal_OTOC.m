%% Thermal OTOC
clear

%% theta=pi/4
figure
load('ther_OTOC_pi4.mat')

time=0:0.001:3;
plot(time,C_T_1,'Color',[0.4940 0.1840 0.5560],'LineWidth',3)
hold on
plot(time,C_T_2,'Color',[0.9290 0.6940 0.1250],'LineWidth',3) 
hold on
plot(time,C_T_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',3)
hold on
plot(time,C_T_4,'Color',[0 0.4470 0.7410],'LineWidth',3)

xlabel('t','fontsize',15);
ylabel('C_T(t)','fontsize',15);
legend('T=100','T=200','T=300','T=400','fontsize',17);
set(gca,'FontSize',20,'linewidth',2.5)  %设置坐标轴字体大小,线粗
axis([0 3 0 55])

