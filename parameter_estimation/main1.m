%% estimate1
% small time interval or strong measurement strength
clear
% diary main1.out %开启日志记录

%% parameter
N = 40;       %total length 
th = 2.0;     %estimated parameter 
lambda = 10.00;   %measurement strength
Num_aver = 1000; %average number

%beta = 0.001测量主导，beta = 0.1演化主导
beta = 0.001;   %small time interval

Tnum = 1000;   %time number
Time = beta*Tnum; 

tic  
[vF_all, vE_sh_all, vE_vn_all, ind_p_all, A_fin_all, B_fin_all] = fn_measure(N,th,lambda,beta,Time,Num_aver); 

endtime = toc()
FileName = ['estimate1'];
save(FileName)


% diary off %关闭日志记录