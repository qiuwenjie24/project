%% Error between FEM and BA, c=100
clear
pos3=[0.11,0.11,0.35 0.37];pos4=[0.51,0.11,0.35 0.37];
pos1=[0.11,0.57,0.35 0.37]; pos2=[0.51,0.57,0.35 0.37]; 
figure

%% eta=1 - even
subplot('position',pos1); 
clearvars -except pos1 pos2 pos3 pos4
load('.\pi4\equal_BA_finite_even_c100.mat','En_even2')
load('Energy_even_no1_c4.mat','En_even','c')

rv_ba = En_even2(1:2000,1);
rv_fem = En_even(1:2000,1);
Eb= sort(rv_ba);
Ef = sort(rv_fem);
% Eb - rv_ba; Ef - rv_fem;

n=2000;
E_dif = Ef(1:n,1) - Eb(1:n,1);
error = E_dif./Eb(1:2000,1);
N_num = [1:1:size(E_dif)]';

% scatter(N_num,error)   % 散点图
plot(N_num,error,'-k','LineWidth',3)

text(50,0.013,'(a) $\eta=1$,even,c=100','fontsize',40,'interpreter','latex')
set(gca,'fontsize',25,'linewidth',3)
% xlabel('n','fontsize',30)
% ylabel('$(E_f-E_b)/E_b$','fontsize',30,'interpreter','latex')

%% eta=1 - odd
subplot('position',pos2); 
clearvars -except pos1 pos2 pos3 pos4
load('.\pi4\equal_BA_finite_odd_c100.mat','En_odd2')
load('Energy_odd_no1_c4.mat','En_odd','c')

rv_ba = En_odd2(1:2000,1);
rv_fem = En_odd(1:2000,1);
Eb= sort(rv_ba);
Ef = sort(rv_fem);
% Eb - rv_ba; Ef - rv_fem;

n=2000;
E_dif = Ef(1:n,1) - Eb(1:n,1);
error = E_dif./Eb(1:2000,1);
N_num = [1:1:size(E_dif)]';

% scatter(N_num,error)   % 散点图
plot(N_num,error,'-k','LineWidth',3)

text(50,0.013,'(b) $\eta=1$,odd,c=100','fontsize',40,'interpreter','latex')
set(gca,'fontsize',25,'linewidth',3)
% xlabel('n','fontsize',30)
% ylabel('$(E_f-E_b)/E_b$','fontsize',30,'interpreter','latex')

%% eta=1/3 - even
subplot('position',pos3); 
clearvars -except pos1 pos2 pos3 pos4
load('.\pi6\BAroots_finite_even_c100.mat','En_even2')
load('Energy_even_no2_c4.mat','En_even','c')

rv_ba = En_even2(1:2000,1);
rv_fem = En_even(1:2000,1);
Eb= sort(rv_ba);
Ef = sort(rv_fem);
% Eb - rv_ba; Ef - rv_fem;

n=2000;
E_dif = Ef(1:n,1) - Eb(1:n,1);
error = E_dif./Eb(1:2000,1);
N_num = [1:1:size(E_dif)]';

% scatter(N_num,error)
plot(N_num,error,'-k','linewidth',3)

text(50,0.013,'(c) $\eta=\frac{1}{3}$,even,c=100','fontsize',40,'interpreter','latex')
set(gca,'fontsize',25,'linewidth',3)
xlabel('n','fontsize',30),ylabel('$(E_f-E_b)/E_b$',...
       'fontsize',30,'interpreter','latex')

%% eta=1/3 - odd
subplot('position',pos4); 
clearvars -except pos1 pos2 pos3 pos4
load('.\pi6\BAroots_finite_odd_c100.mat','En_odd2')
load('Energy_odd_no2_c4.mat','En_odd','c')

rv_ba = En_odd2(1:2000,1);
rv_fem = En_odd(1:2000,1);
Eb= sort(rv_ba);
Ef = sort(rv_fem);
% Eb - rv_ba; Ef - rv_fem;

n=2000;
E_dif = Ef(1:n,1) - Eb(1:n,1);
error = E_dif./Eb(1:2000,1);
N_num = [1:1:size(E_dif)]';

% scatter(N_num,error)
plot(N_num,error,'-k','linewidth',3)

text(50,0.013,'(d) $\eta=\frac{1}{3}$,odd,c=100','fontsize',40,'interpreter','latex')
set(gca,'fontsize',25,'linewidth',3)
