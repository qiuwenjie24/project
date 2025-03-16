%% Nearest-neighbor spacing distribution c=infity
clear

%% pi/4
load('triangle_pi_4.mat','rv')

E_n=sortrows(rv,1) ;    % 排序
level_sp = E_n(2:end) - E_n(1:end-1);  % 能级间隔
level_sp_new = sortrows(level_sp,1);  %排序

jj1 = floor( 0.99* max(size(level_sp)) );  % 向下取整，人为调整参数
level_sp_new(jj1:end) = [];  % 删除太大的能级间隔

%调节jj1确保level_sp_new/level_sp_aver大部分能级落在edges区间中，并忽略区间外的点
level_sp_aver = sum(level_sp_new)/max(size(level_sp_new));    %average level spacing

%% Fig
figure
edges = 0:0.1:4;   % 区间划分

h2=histogram(level_sp_new./level_sp_aver,edges);    
h2.Normalization = 'pdf';   %频数变频率密度
xlabel('s/s_{aver}')
ylabel('NSD')

hold on
plot(edges,fn_Brody(edges,0),'o-k')  % poisson
hold on
plot(edges,fn_Brody(edges,1),'o-b')  % wigner