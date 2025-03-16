% shanon entropy, Fisher information AAH
% A_fin_all采用qr分解的归一, B_fin_all直接归一
function [vF_all, vE_sh_all, vE_vn_all, ind_p_all, A_fin_all, B_fin_all] = fn_measure(N,th,lambda,beta,Time,Num_aver)

%% parameter
Dis_str = 0;  %disorder strength
Cp = 1;       %coupling constant

M = N/2;      %total particle number
tnum = round(Time/beta);  %time number
% Num_aver = 300;  %average times of random

%% Hamitonian and evolution operator
[~, H_op] = fn_AAH(N, Cp, Dis_str);
H = H_op;

[H_s,H_d] = eig(H);   %norm(H - H_s* H_d *H_s' )
F1 = diag(H_d);
F2 = exp(-1i*beta*F1); F3 = diag(F2); 
F4 = exp(-1i*th*F1); F5 = diag(F4); 

U_be = H_s*F3*H_s';   %evolutin operator beta
U_th = H_s*F5*H_s';   %encoding operator theta

%% effective measure matrix
% Mea = cell(1, N);
% for n = 1:N
%    tmp = eye(N);
%    tmp(n, n) = exp(-lambda/2);     %对第n个格点的弱测量
%    Mea{n} = tmp;                    
% end


%% ------------------parfor--------------------------------
ind_p_all = zeros(Num_aver, tnum);    
vF_all = zeros(Num_aver, tnum);       
vE_sh_all = zeros(Num_aver, tnum);
vE_vn_all = zeros(Num_aver, tnum); 
A_fin_all = cell(Num_aver, 1); 
B_fin_all = cell(Num_aver, 1); 
parfor jj2=1:Num_aver

%--------------------effective measure matrix---------------------
%放在并行内时为了避免Mea变成广播变量(每个循环体都调用)，造成不必要的通信开销
Mea = cell(1, N);
norm_tmp = sqrt( N + (exp(-lambda) - 1)*M );   %测量算符的系数，sum On=1
for n = 1:N
   tmp = eye(N);
   tmp(n, n) = exp(-lambda/2);     %对第n个格点的弱测量
   Mea{n} = tmp;
end
%--------------------effective measure matrix---------------------
    
% dis_vP = zeros(N, tnum);  
ind_p = zeros(1, tnum);    
vF = zeros(1, tnum);       
vE_sh = zeros(1, tnum);
vE_vn = zeros(1, tnum); 

A = eye(N);       %matrix of initial state
B = eye(N);
for jj1=1:tnum
    
    if jj1>=2
        A = U_be*A;       %time evolutin
        B = U_be*B;
    else
        A = U_th*A;       %encoding
        B = U_th*B;
    end
    
    vP = fn_P(A, lambda, M);           %probability 1*N
    vP_pd = fn_P_pd(A, lambda, M);     %derivative of probability 1*N
%     dis_vP(:,jj1) = transpose(vP);       %probability distribution
    
    vF(jj1) = fn_Fisher(vP, vP_pd);    %classical fisher information
    vE_sh(jj1) = fn_Sh_Entropy(vP);    %shanon entropy log2
    vE_vn(jj1) = fn_vN_Entropy(A,M);   %von Neumann entropy
    
    p = fn_rand_ind(vP);              %result index p
    ind_p(jj1) = p;                   %index of measurement result
    A = Mea{p}*A./( norm_tmp^(1/M) ); %matrix after measurement
    A = A./( (sqrt(vP(p))).^(1/M) );   %simple normalization
    
    B = Mea{p}*B./( norm_tmp^(1/M) );
    [Q,R] = qr(B);                    %qr normalization
    B = Q;
    

end

A_fin_all{jj2} = A;
B_fin_all{jj2} = B;
ind_p_all(jj2, :) = ind_p;    
vF_all(jj2, :) = vF;       
vE_sh_all(jj2, :) = vE_sh;
vE_vn_all(jj2, :) = vE_vn; 

% fprintf('%d end \n',jj2)
end


end
