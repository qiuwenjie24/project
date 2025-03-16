%% Error between FEM and BA
pos1=[0.11,0.23,0.35 0.57]; pos2=[0.57,0.23,0.35 0.57];
figure

%% pi/4 
clearvars -except pos1 pos2
load('BA_pi4.mat','root_ba')
load('triangle_pi_4.mat','high','rv')

rv_ba = root_ba(:,3);
rv_fem = rv;
Eb= sort(rv_ba);
Ef = sort(rv_fem);
clearvars -except pos1 pos2 Eb Ef high

n=2000;
E_dif = Ef(1:n,1) - Eb(1:n,1);
error = E_dif./Eb(1:2000,1);
N_num = [1:1:size(E_dif)]';

subplot('position',pos1);
% scatter(N_num,error)   % 散点图
plot(N_num,error,'-k','LineWidth',3)

text(50,0.0018,'(a) $\eta=1$','fontsize',40,'interpreter','latex')
set(gca,'fontsize',30,'linewidth',2)
xlabel('n','fontsize',30),ylabel('$(E_f-E_b)/E_b$',...
       'fontsize',30,'interpreter','latex')

%% pi/6 
clearvars -except pos1 pos2
load('BA_pi6.mat','root_ba')
load('triangle_pi_6.mat','high','rv')

rv_ba = root_ba(:,3);
rv_fem = rv;
Eb= sort(rv_ba);
Ef = sort(rv_fem);
clearvars -except pos1 pos2 Eb Ef high

n=2000;
E_dif = Ef(1:n,1) - Eb(1:n,1);
error = E_dif./Eb(1:2000,1);
N_num = [1:1:size(E_dif)]';

subplot('position',pos2);
% scatter(N_num,error)
plot(N_num,error,'-k','linewidth',3)

text(50,0.0027,'(b) $\eta=\frac{1}{3}$','fontsize',40,'interpreter','latex')
set(gca,'fontsize',30,'linewidth',2)
