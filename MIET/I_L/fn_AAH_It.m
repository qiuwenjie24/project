%% AAH  It  并行
% L is even
function [Ig_all, VarIg_all, index] = fn_AAH_It(L, gam, Dis_str, aver_num, tnum, phase)
%% parameter
% L = 800;        %system size
% gam = 0.1;      %measurement strength
% Dis_str = 0;    %disorder strength
% aver_num = 25;  %average times of random 200
% tnum = 3000;    %time number, T=tnum*Dt
% phase=0;        %disorder phase

N = L/2;     %total particle number
Dt = 0.05;   %time step
Cp = 1;      %coupling constant

lA = L/8;      %partion length A
lB = lA;       %partion length B
rAB = L/2;     %central distance of partion A and B
Am1 = 1;       %partion A of [Am1,Am2]
Am2 = Am1 + lA - 1;
Bm1 = Am1 + rAB;      %partion B of [Bm1,Bm2]
Bm2 = Bm1 + lB - 1;

%% Hamitonian H_fr and evolution operator Exp_h
v = ones(L-1,1);
H_fr = diag(v,1) + diag(v,-1);   %open boundary
H_fr(1,L) = 1;  H_fr(L,1) = 1;   %periodic boundary
H_fr = H_fr.*Cp;

v1 = (1:L);%randn(1,L);
beta = (sqrt(5)-1 )/2;
Dis_coff = cos(2.*pi.*beta.*v1 + phase);
Dis_term = diag(Dis_coff);  %disorder term
H_fr = H_fr + Dis_str.*Dis_term;

% Exp_h = expm(-1i*Dt*H_fr);  %evolutin operator, but expm() slower
[H_s,H_d] = eig(H_fr);   %H_fr=H_s* H_d *H_s'
F1 = diag(H_d);
F2 = exp(-1i*Dt*F1);
F3 = diag(F2);
Exp_h = H_s*F3*H_s';    %expm(-1i*Dt*H_fr)=Exp_h

%% initial state U_init, correlation matrix D_init
U_init = zeros(L); V0 = eye(L); ii1 = 1;
for ii2=2:2:L   %L is even
    U_init(:,ii1) = V0(:,ii2);
    U_init(:,L-ii1+1) = V0(:,ii2-1);
    ii1 = ii1 + 1;
end

U_temp = U_init(:,1:N); %U_init is L*L, U_temp is L*N
D_init = transpose(U_temp * U_temp');  %correlation matrix of Neel state L*L

% U_nn = U_init(:,N+1:L);
% D_nn = (U_nn * U_nn');

%%
Se_all_A = zeros(tnum,aver_num);
Se_all_B = zeros(tnum,aver_num);
Se_all_AB = zeros(tnum,aver_num);
VarSl_all_A = zeros(tnum,aver_num);
VarSl_all_B = zeros(tnum,aver_num);
VarSl_all_AB = zeros(tnum,aver_num);
parfor jj3=1:aver_num   % average evolution trajectory
    %% 预生成当前平均次的随机数
    rng('shuffle')
    mu = 0; sigma = gam*Dt;  %均值，方差
    R_num = sqrt(sigma).*randn(L,tnum) + mu;  %random number L*tnum
    U0 = U_init;
    D0 = D_init;
    for jj1=1:tnum    % evolution time
        %% calculate evolution of D0 and U0
        vv = R_num(:,jj1) + ( 2.*diag(D0) - 1).*gam.*Dt;
        M = diag(exp(vv));
        V = M* (Exp_h * U0);
        [Q,~] = qr(V);  %QR 分解, V = Q*R ; Q'*Q=I
        
        U0 = Q;
        U_temp = U0(:,1:N);
        D0 = transpose(U_temp * U_temp');
        
        %% 每5个点往下计算一次物理量
        if jj1 ~= 1 && mod(jj1,5) ~= 0
            continue
        end
        
        %% entanglment entropy Se_all_A, Se_all_B, Se_all_AB
        D_sub = D0;
        %(可能是)精度问题导致有时候对角化会失败,若失败则尝试调节精度
%         D_sub = round(D_sub, 9);  %调节精度，四舍五入到小数点后9位,但这步似乎有点耗时间
        
        D_sub_A = D_sub(Am1:Am2,Am1:Am2);  %D_sub_A is lA*lA
        Lam_A = eig(D_sub_A, 'vector');
        Lam_A(abs(Lam_A)<eps) = eps;
        Lam_A(abs(Lam_A-1)<eps) = 1 - eps;
        SS_A = Lam_A.*log2(Lam_A) + (1-Lam_A).*log2(1-Lam_A);
        Se_all_A(jj1,jj3) = -1*sum(SS_A);
        
        D_sub_B = D_sub(Bm1:Bm2,Bm1:Bm2);  %D_sub_B is lB*lB
        Lam_B = eig(D_sub_B, 'vector');
        Lam_B(abs(Lam_B)<eps) = eps;
        Lam_B(abs(Lam_B-1)<eps) = 1 - eps;
        SS_B = Lam_B.*log2(Lam_B) + (1-Lam_B).*log2(1-Lam_B);
        Se_all_B(jj1,jj3) = -1*sum(SS_B);
        
        D_sub_AB = [D_sub(Am1:Am2,Am1:Am2), D_sub(Am1:Am2,Bm1:Bm2);
            D_sub(Bm1:Bm2,Am1:Am2), D_sub(Bm1:Bm2,Bm1:Bm2)];
        %D_sub_AB is (lA+lB)*(lA+lB)
        Lam_AB = eig(D_sub_AB, 'vector');
        Lam_AB(abs(Lam_AB)<eps) = eps;
        Lam_AB(abs(Lam_AB-1)<eps) = 1 - eps;
        SS_AB = Lam_AB.*log2(Lam_AB) + (1-Lam_AB).*log2(1-Lam_AB);
        Se_all_AB(jj1,jj3) = -1*sum(SS_AB);
        
        %% fluctuation VarSl_A VarSl_B VarSl_AB
        U_nn = U0(:,N+1:L);
        D_nn = (U_nn * U_nn');
        vv1 = D0 .*D_nn;
        vv2_A = vv1(Am1:Am2,Am1:Am2);
        vv2_B = vv1(Bm1:Bm2,Bm1:Bm2);
        vv2_AB = [vv1(Am1:Am2,Am1:Am2), vv1(Am1:Am2,Bm1:Bm2);
            vv1(Bm1:Bm2,Am1:Am2), vv1(Bm1:Bm2,Bm1:Bm2)];
        
        VarSl_all_A(jj1,jj3) = sum(vv2_A,'all');  %tnum*aver_sum
        VarSl_all_B(jj1,jj3) = sum(vv2_B,'all');
        VarSl_all_AB(jj1,jj3) = sum(vv2_AB,'all');
        
    end
    
    fprintf('%d end \n',jj3)
end

%% saturation Ig_sat, transientIg_t
Ig_all = Se_all_A + Se_all_B - Se_all_AB;  %tnum*aver_sum

%% saturation VarSl_sat, transient VarSl_t
VarIg_all = VarSl_all_A + VarSl_all_B - VarSl_all_AB;  %tnum*aver_sum

%% 记录计算物理量的时间点
index = [1, 5:5:tnum];
Ig_all = Ig_all(index,:);
VarIg_all = VarIg_all(index,:);

end

