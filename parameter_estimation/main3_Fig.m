%% figure-S3
close all
clear
%% check
load('estimate2.mat')
load('estimate3.mat')
errer0 = abs(th_sim - th_sim_old0);
errer1 = abs(th_sim - th_sim_old1);
if max(errer0)>0.2
    fprintf('maybe th_sim_old0 and th_sim are wrong!!\n')
    %说明似然分布函数不是很光滑，或许是数值本身误差造成的
end
if max(errer1)>5*1e-4 
    fprintf('maybe th_sim_old1 and th_sim are wrong!!\n')
    %出现这种情况一般是步长过大导致的
end
a0 = sort(errer0.','descend');
a1 = sort(errer1.','descend');
errer2 = abs(th_sim_old0 - th_sim_old1);
a2 = sort(errer2.','descend');

%% average value of all th 
clear
FileName = ['estimate3'];
load(FileName)

th_sim_new = th_sim;
% th_sim_new(th_sim_new<1.5+0.0001)=[];  %排除过于偏离范围的估计值
% th_sim_new(th_sim_new>2.5-0.0001)=[];
num_new = max(size(th_sim_new));
th_sim_aver_new = zeros(1,num_new);

for jj2 = 1:num_new
    th_sim_aver_new(jj2) = mean(th_sim_new(1:jj2),'all');
end

figure
plot(1:num_new,th_sim_aver_new,'o-')
hold on 
plot( [1,num_new], [2,2],'--k','LineWidth',2)
xlabel('average number n')
ylabel('average estimator $\bar{\theta}_{es}(n)$','interpret','latex')
title('S--$\lambda = 10,\beta = 0.001$','interpret','latex')

%% the th value with largest number in all th 
clear
FileName = ['estimate3'];
load(FileName)

th_sim_rd = round(th_sim,1);  %0.1
num_rd = max(size(th_sim_rd));
th_sim_opt = zeros(1,num_rd);  %largest number th 

th_sim_mean = zeros(1,num_rd);
for jj2 = 1:num_rd
    th_sim_opt(jj2) = mode(th_sim_rd(1:jj2),'all');
    
    temp1 = th_sim_opt(jj2);
    temp2 = th_sim(1:jj2);
    temp3 = temp2(temp1-0.05 <= temp2 & temp2 < temp1+0.05);
    th_sim_mean(jj2) = mean(temp3,'all');
end

figure
plot(1:num_rd,th_sim_mean,'o-')
xlabel('number m of evolution trajetories')
ylabel('estimated value $\bar{\theta}_{es}(n)$','interpret','latex')
title('S--$\lambda = 10,\beta = 0.001$','interpret','latex')


%% distribution of th
clear
FileName = ['estimate3'];
load(FileName)
figure
edges = [-0.9:0.2:3.7];
h = histogram(th_sim, edges, 'Normalization','probability');
xlabel('$\theta_{es}$','interpret','latex')
ylabel('Probability')
title('S--$\lambda = 10,\beta = 0.001$','interpret','latex')

