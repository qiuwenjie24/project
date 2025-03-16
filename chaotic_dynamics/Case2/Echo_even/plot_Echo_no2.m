%% echo - integrable 2 all
clear
figure  
pos3=[0.11,0.11,0.35 0.37];pos4=[0.51,0.11,0.35 0.37];
pos1=[0.11,0.57,0.35 0.37]; pos2=[0.51,0.57,0.35 0.37]; 

%%
load('data_echo_no2_c1.mat','T1','inte_real1')
subplot('position',pos1); 
plot(T1,inte_real1,'b-','linewidth',4) 
hold on
clearvars -except pr pb pk pg pos1 pos2 pos3 pos4
set(gca,'FontSize',20)
text(0.01,0.8,'(a) $\eta=\frac{1}{3},c=0$','interpreter','latex','fontsize',30); 
set(gca,'linewidth',2.5); %坐标轴 线粗0.5磅

%%
load('data_echo_no2_c2.mat','T1','inte_real1')
subplot('position',pos2); 
plot(T1,inte_real1,'b-','linewidth',4) %,axis([0 TT 0 1.3])
hold on
clearvars -except pr pb pk pg pos1 pos2 pos3 pos4
set(gca,'FontSize',20)
text(0.01,0.8,'(b) $\eta=\frac{1}{3},c=1$','interpreter','latex','fontsize',30); 
set(gca,'linewidth',2.5); %坐标轴 线粗0.5磅

%%
load('data_echo_no2_c3.mat','T1','inte_real1')
subplot('position',pos3); 
plot(T1,inte_real1,'b-','linewidth',4) 
hold on
clearvars -except pr pb pk pg pos1 pos2 pos3 pos4
set(gca,'FontSize',20)
xlabel('$t$','interpreter','latex','fontsize',30)
ylabel('$||\langle \psi_e|U_c(t)|\psi_e \rangle||^2$',...
         'interpreter','latex','fontsize',30)
text(0.01,0.8,'(c) $\eta=\frac{1}{3},c=10$','interpreter','latex','fontsize',30); 
set(gca,'linewidth',2.5); %坐标轴 线粗0.5磅

%%
load('data_echo_no2_c4.mat','T1','inte_real1')
subplot('position',pos4); 
plot(T1,inte_real1,'b-','linewidth',4) 
hold on
clearvars -except pr pb pk pg pos1 pos2 pos3 pos4
set(gca,'FontSize',20)
text(0.01,0.8,'(d) $\eta=\frac{1}{3},c=100$','interpreter','latex','fontsize',30); 
set(gca,'linewidth',2.5); %坐标轴 线粗0.5磅