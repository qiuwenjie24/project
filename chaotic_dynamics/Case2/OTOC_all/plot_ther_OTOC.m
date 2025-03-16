%% ther OTOC - integrable 2 c1-c4
clear
figure  
pos3=[0.11,0.11,0.35 0.37];pos4=[0.51,0.11,0.35 0.37];
pos1=[0.11,0.57,0.35 0.37]; pos2=[0.51,0.57,0.35 0.37]; 

%% 
subplot('position',pos1);
load('.\no2\c1\ther_OTOC_all_no2_c1.mat')  

time=0:0.001:3;
plot(time,C_T_1,'Color',[0.4940 0.1840 0.5560],'LineWidth',3)
hold on
plot(time,C_T_2,'Color',[0.9290 0.6940 0.1250],'LineWidth',3) 
hold on
plot(time,C_T_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',3)
hold on
plot(time,C_T_4,'Color',[0 0.4470 0.7410],'LineWidth',3)

text(0.1,150,'(a) $\eta=\frac{1}{3},c=0$','interpreter','latex','fontsize',30)
set(gca,'FontSize',20,'linewidth',2.5)  %设置坐标轴字体大小,线粗
axis([0,3,0,165])

%% 
subplot('position',pos2);
load('.\no2\c2\ther_OTOC_all_no2_c2.mat')  

time=0:0.001:3;
plot(time,C_T_1,'Color',[0.4940 0.1840 0.5560],'LineWidth',3)
hold on
plot(time,C_T_2,'Color',[0.9290 0.6940 0.1250],'LineWidth',3) 
hold on
plot(time,C_T_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',3)
hold on
plot(time,C_T_4,'Color',[0 0.4470 0.7410],'LineWidth',3)

text(0.1,150,'(b) $\eta=\frac{1}{3},c=1$','interpreter','latex','fontsize',30)
set(gca,'FontSize',20,'linewidth',2.5)  %设置坐标轴字体大小,线粗
axis([0,3,0,165])

%% 
subplot('position',pos3);
load('.\no2\c3\ther_OTOC_all_no2_c3.mat')  

time=0:0.001:3;
plot(time,C_T_1,'Color',[0.4940 0.1840 0.5560],'LineWidth',3)
hold on
plot(time,C_T_2,'Color',[0.9290 0.6940 0.1250],'LineWidth',3) 
hold on
plot(time,C_T_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',3)
hold on
plot(time,C_T_4,'Color',[0 0.4470 0.7410],'LineWidth',3)

text(0.1,80,'(c) $\eta=\frac{1}{3},c=10$','interpreter','latex','fontsize',30)
set(gca,'FontSize',20,'linewidth',2.5)  %设置坐标轴字体大小,线粗
xlabel('t','fontsize',25),ylabel('C_T(t)','fontsize',25);
legend('T=100','T=200','T=300','T=400','fontsize',17);
axis([0,3,0,90])

%% 
subplot('position',pos4);
load('.\no2\c4\ther_OTOC_all_no2_c4.mat')  

time=0:0.001:3;
plot(time,C_T_1,'Color',[0.4940 0.1840 0.5560],'LineWidth',3)
hold on
plot(time,C_T_2,'Color',[0.9290 0.6940 0.1250],'LineWidth',3) 
hold on
plot(time,C_T_3,'Color',[0.8500 0.3250 0.0980],'LineWidth',3)
hold on
plot(time,C_T_4,'Color',[0 0.4470 0.7410],'LineWidth',3)

text(0.1,80,'(d) $\eta=\frac{1}{3},c=100$','interpreter','latex','fontsize',30)
set(gca,'FontSize',20,'linewidth',2.5)  %设置坐标轴字体大小,线粗
axis([0,3,0,90])