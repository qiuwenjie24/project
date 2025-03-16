%% entanglement entropy of half chain
%state matrix-A, particle number-M 
function res = fn_vN_Entropy(A,M)

[N, ~] = size(A);  %length

cor_D = fn_Cor(A,M)./fn_OverLap(A,A,M);    %D_ij = <a+_j a_i> N*N
% A_new = A(:,1:M);   %这种得到cor_D的方法只适用于A是幺正的情况
% cor_D = transpose(A_new*A_new');

cor_D_sub = cor_D(1:N/2,1:N/2);           %half chain
cor_D_sub = round( (1e+9)*cor_D_sub ) /(1e+9); %可能是精度问题导致有时候对角化失败
    
[~,Dsub_d] = eig(cor_D_sub);
D_ev = diag(Dsub_d);
D_ev(abs(D_ev)<10^(-8))=0; 
D_ev(abs(D_ev-1)<10^(-8))=1;

tep = D_ev.*log(D_ev) + (1-D_ev).*log(1-D_ev);
tep(isnan(tep)) = 0;    %使用isnan函数检查NaN值
vE_vn = -1*sum(tep);    %von Neumann entropy
res = vE_vn;           

end