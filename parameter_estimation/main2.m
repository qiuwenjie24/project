%% estimate2
clear
% diary main2.out %开启日志记录

FileName = ['estimate1'];
load(FileName,'A_fin_all','th','N','Tnum')

Cp = 1; Dis_str = 0; M = N/2;
%% Hamitonian

[~, H_op] = fn_AAH(N, Cp, Dis_str);
H = H_op;

[H_s,H_d] = eig(H);   %norm(H - H_s* H_d *H_s' )
vep = diag(H_d);
F4 = exp(-1i*th*vep); F5 = diag(F4); 
U_th = H_s*F5*H_s';   %encoding operator theta

%%
error = 2.5; num_th = 101;
vaule_th = linspace(2-error,2+error,num_th);

num_ave = max(size(A_fin_all));
th_sim_old0 = zeros(1,num_ave);
tic
parfor jj1 = 1:num_ave

A = A_fin_all{jj1};
Af = A* U_th';

max_th = fn_find_max_old0(Af, M, H_s, vep, vaule_th);

th_sim_old0(jj1) = max_th;

% fprintf('jj1 = %d, max_th = %d, \n',jj1, max_th)
end

end_time = toc();
FileName =  ['estimate2'];
save(FileName,'th_sim_old0','end_time','vaule_th')

% diary off %关闭日志记录