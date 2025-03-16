%% AAH  Slogl-t  并行
% L is even
function [Se_sub_all, VarSl_sub_all, index] = fn_AAH_SLsub(L, gam, Dis_str, aver_num, tnum, phase, Lsub_all)
%% parameter
% L = 800;        %system size
% gam = 0.1;      %measurement strength
% Dis_str = 0;    %disorder strength
% aver_num = 25;  %average times of random 200
% tnum = 3000;    %time number, T=tnum*Dt
% phase=0;        %disorder phase

% Lsub_all = [100,200];  %subsystem size
N = L/2;     %total particle number
Dt = 0.05;   %time step
Cp = 1;      %coupling constant

%% Hamitonian H_fr and evolution operator Exp_h
v = ones(L-1,1);
H_fr = diag(v,1) + diag(v,-1);   %open boundary
H_fr(1,L) = 1;  H_fr(L,1) = 1;   %periodic boundary
H_fr = H_fr.*Cp;

v1 = (1:L);
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

%%
Lsub_num = max(size(Lsub_all));
Se_sub_all = zeros(tnum,aver_num, Lsub_num);
VarSl_sub_all = zeros(tnum,aver_num, Lsub_num);
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
        
        %% 计算所有子系统
        Lsub_all_temp = Lsub_all;   %Lsub_all是广播变量，Lsub_all_temp是局部变量，后者更利于并行
        for jj_L = 1:Lsub_num
            Lsub = Lsub_all_temp(jj_L);
            %% entanglment entropy Se between [0, Lsub] and [Lsub+1, L]
            D_sub = D0(1:Lsub,1:Lsub); %D_sub is Lsub*Lsub
            %(可能是)精度问题导致有时候对角化会失败,若失败则尝试调节精度
            %D_sub = round(D_sub, 9);  %调节精度，四舍五入到小数点后9位,但这步似乎有点耗时间            
            Lam = eig(D_sub, 'vector');    %[~,Lam] = eig(D_sub)更慢，两种计算方式不同，数值结果稍许不同
            
            Lam(abs(Lam)<eps) = eps;
            Lam(abs(Lam-1)<eps) = 1 - eps;
            SS = Lam.*log2(Lam) + (1-Lam).*log2(1-Lam);
            Se_sub_all(jj1,jj3,jj_L) = -1*sum(SS);
            
            %% fluctuation VarSl
            U_nn = U0(:,N+1:L);
            D_nn = U_nn * U_nn';
            vv2 = D0(1:Lsub,1:Lsub) .*D_nn(1:Lsub,1:Lsub);
            VarSl_sub_all(jj1,jj3,jj_L) = sum(vv2,'all');  %tnum*aver_sum
        end
    end
    
    fprintf('%d end \n',jj3)
end

%% 记录计算物理量的时间点
index = [1, 5:5:tnum];
Se_sub_all = Se_sub_all(index,:,:);
VarSl_sub_all = VarSl_sub_all(index,:,:);

end  %function end
