%% estimate3
clear
% diary main3.out %开启日志记录

FileName = ['estimate1'];
load(FileName,'A_fin_all','th','N')

Cp = 1; Dis_str = 0; M = N/2;
%% Hamitonian

[~, H_op] = fn_AAH(N, Cp, Dis_str);
H = H_op;

[H_s,H_d] = eig(H);   %norm(H - H_s* H_d *H_s' )
vep = diag(H_d);
F4 = exp(-1i*th*vep); F5 = diag(F4); 
U_th = H_s*F5*H_s';   %encoding operator theta

%%
tic
num_ave = max(size(A_fin_all));
th_sim = zeros(1,num_ave);
th_sim_old1 = zeros(1,num_ave);

load('estimate2.mat','th_sim_old0')
parfor jj1 = 1:num_ave

% th0 = 1*th;   %any initial value th
th0 = th_sim_old0(jj1);  %initial value th from estimate_S2.mat

A = A_fin_all{jj1};
Af = A* U_th';

max_th = fn_find_max(Af, M, H_s, vep, th0);
max_th_old1 = fn_find_max_old1(Af, M, H_s, vep, th0);
% fn_find_max 是通过求该点的斜率的最速下降法
% fn_find_max_old 是通过求该附近的点得到的最速下降法
% if abs(max_th-max_th_old1)>1e-3  %check fn_find_max_old
%     fprintf('error is %d \n',abs(max_th-max_th_old1))
% end

th_sim(jj1) = max_th;
th_sim_old1(jj1) = max_th_old1;

fprintf('jj1 = %d, max_th = %d, time=%s \n',jj1, max_th, datetime)
end


end_time = toc();
FileName = ['estimate3'];
save(FileName,'th','th_sim','th_sim_old1','end_time')


% diary off %关闭日志记录