% particle probility, measurement probility, and some checks
% A采用直接归一, B直接qr分解的归一
% small time interval or strong measurement strength
clear
%% parameter
N = 40;       %total length 
M = N/2;      %total particle number
Dis_str = 0;  %disorder strength
Cp = 1;       %coupling constant
th = 2.0;     %estimated parameter 
lambda = 10.00;  %measurement strength

%beta = 0.001测量主导，beta = 0.1演化主导
beta = 0.001;  %time interval

Tnum = 1001;  %time number
Time = Tnum*beta;
% Num_aver = 1;  %average times of random

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
Mea = cell(1, N);
norm_tmp = sqrt( N + (exp(-lambda) - 1)*M );   %归一化系数后面作用时再加进去
for n = 1:N
   tmp = eye(N);
   tmp(n, n) = exp(-lambda/2);     %对第n个格点的弱测量
   Mea{n} = tmp;
end


%% ------------------parfor--------------------------------
    
% dis_vP = zeros(N, tnum);  
ind_p = zeros(1, Tnum);    
vF = zeros(1, Tnum);       
vE_sh = zeros(1, Tnum);
vE_vn = zeros(1, Tnum); 
vE_vn_B = zeros(1, Tnum); 

A = eye(N);       %matrix of initial state 
B = eye(N);
A_p = eye(N);  lr=1e-6;
for jj1=1:Tnum
    
    if jj1>=2
        A = U_be*A;       %time evolutin
        B = U_be*B;
        A_p = U_be*A_p;
    else
        A = U_th*A;       %encoding
        B = U_th*B; 
        A_p = H_s*diag(exp(-1j*(th+lr)*F1))*H_s'*A_p;
    end
    
%---------------check1------------------------
    %%---验证A 与 A_p
%     tep8 = norm(A*U_th'*H_s*diag(exp(-1j*(th+lr)*F1))*H_s' - A_p) %norm(U_th*U_th')*1e5
%     if tep8>1e-6
%         fprintf('error \n')
%     end
    %%---验证fn_OverLap，fn_OverLap_pd
%     OverLap = fn_OverLap(A,A,M);
%     OverLap_P = (fn_OverLap(A_p,A_p,M));
%     A_ini = eye(N);
%     tep3 = fn_OverLap_pd(A_ini, A, M);
%     tep4 = ( (OverLap_P - OverLap)/lr );
%     tep5 = fn_OverLap_pd(A_ini, A_p, M) ;
%     abs(tep3-tep4)
%     if abs(tep3-tep4)>1e-5 || abs(tep4-tep5)>1e-5
%         fprintf('error is %d \n',abs(tep3-tep4))
%     end
%--------------check1-------------------------
    
    vP = fn_P(A, lambda, M);           %probability 1*N
    vP_B = fn_P(B, lambda, M);
%--------------check2-------------------------
    %%----验证QR分解和直接分解都给出一样概率
%     vP_err = vP_B - vP;
%     norm(vP_err)
%     if norm(vP_err)>1e-6
%        fprintf('%d-%d error of vP is %d \n',jj2,jj1,norm(vP_err))
%     end
%--------------check2-------------------------

    vP_pd = fn_P_pd(A, lambda, M);  %derivative of probability 1*N
%     dis_vP(:,jj1) = transpose(vP);       %probability distribution
    
    vF(jj1) = fn_Fisher(vP, vP_pd);    %classical fisher information
    vE_sh(jj1) = fn_Sh_Entropy(vP);    %shanon entropy
    vE_vn(jj1) = fn_vN_Entropy(A,M);   %von Neumann entropy
    vE_vn_B(jj1) = fn_vN_Entropy(B,M);
%--------------check3-------------------------
        %%----验证QR分解和直接分解都给出一样纠缠熵
%     vE_vn_err = vE_vn(jj1) - vE_vn_B(jj1);
%     norm(vE_vn_err)
%     if norm(vE_vn_err)>1e-6
%        fprintf('%d-%d error of vE_vn is %d \n',jj2,jj1,norm(vE_vn_err))
%     end
%--------------check3-------------------------
    
    p = fn_rand_ind(vP);              %result index p
    ind_p(jj1) = p;                   %index of measurement result
    A = Mea{p}*A./( norm_tmp^(1/M) ); %matrix after measurement
    A = A./( (sqrt(vP(p))).^(1/M) );   %simple normalization
    
    B = Mea{p}*B./( norm_tmp^(1/M) );
    [Q,R] = qr(B);                    %qr normalization
    B = Q;
    
    A_p = Mea{p}*A_p./( norm_tmp^(1/M) );
    A_p = A_p./( (sqrt(vP(p))).^(1/M) );
    
    
%--------------------check4-------------------
%     tep1 = diag(R); tep2 = prod( tep1(1:M) ,'all');
%     p_error = conj(tep2)*tep2 - vP(p);
%     if norm(p_error)>1e-8
%        %验证qr分解给的归一化系数和直接归一化是相同的
%        %加入了norm_tmp，使得det(R(1:M,1:M))为态的归一化系数
%        %注意qr分解归一化是找等效一个A矩阵，与直接归一得到的A不一样
%        fprintf('%d-%d error of vP is %d \n',jj2,jj1,p_error)
%     end
%--------------------check4-------------------

%--------------------check2-------------------
%     %验证QR归一化方法
%     B_norm = fn_OverLap(B,B,M);
%     if norm(B_norm - 1)>1e-6
%        fprintf('%d-%d error of QR normalization is %d \n',jj2,jj1,norm(B_norm -1))
%     end
%     %验证直接归一化方法
%     A_norm = fn_OverLap(A,A,M)
%     if norm(A_norm - 1)>1e-6
%        fprintf('%d-%d error of simple normalization is %d \n',jj2,jj1,norm(A_norm -1))
%     end
%--------------------check2-------------------

    figure(8); 
    plot(vP,'-bo'); 
    hold on; 
%     plot(p, vP(p), 'r*'); 
    hold off;
    xlabel('position [1,...,N]')
    ylabel('measurement probability $[P_1,...,P_N]$','interpret','latex')
%     title('t=0')
    title(['t=',num2str(beta*(jj1-1)),',  $\beta$=',num2str(beta)],...
           'interpret','latex')
    axis([0 N 0 0.05])
    
    Cor_ada = (vP* (N + (exp(-lambda)-1)*M) - 1 )/(exp(-lambda)-1);  %<a+ a>
    Cor_ada(abs(Cor_ada)<1e-8) = 0;
    Cor_ada(abs(Cor_ada-1)<1e-8) = 1;
    figure(9); 
    plot(Cor_ada,'-bo'); 
    xlabel('position [1,...,N]')
    ylabel('particle probability','interpret','latex')
%     title('t=0')
    title(['t=',num2str(beta*(jj1-1)),',  $\beta$=',num2str(beta)],...
           ['$t/\beta$=',num2str((jj1-1))],'interpret','latex')
    axis([0 N 0 1])
    pause(0.0001);
    
%     if jj1==Tnum
%        FileName = ['..\data_origin\PD_Ys.xlsx']
%        xlswrite(FileName,Cor_ada')
%     elseif jj1==1
%        FileName = ['..\data_origin\PD_Y.xlsx']
%        xlswrite(FileName,Cor_ada')
%     end
end


% figure; 
% plot(vE_vn,'-bo'); 
% figure; 
% plot(vE_vn_B,'-bo'); 
