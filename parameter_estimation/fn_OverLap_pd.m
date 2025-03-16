%partial_th <Psi_A| Psi_A>
%初始A0，现在At，总长度N, 粒子数M
function res = fn_OverLap_pd(A0, At, M)

B0 = A0;
B0(:,[M,M+1]) = B0(:,[M+1,M]);
Bt = At*B0;

[Q,R] = qr(At);
tep1 = diag(R);  %N*1
tepA = prod( tep1(1:M) ,'all'); 

C = Q'*Bt;

OverLap_pd1 = (1j)*tepA*conj(det(C(1:M,1:M)));  %<Bt|At>
OverLap_pd2 = conj(OverLap_pd1);                %<At|Bt>

res = OverLap_pd1 + OverLap_pd2;
end




