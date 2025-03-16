%% find th of maximum probability by SDM  更快
%state matrix-A=Af*U_th, particle number-M 
function res = fn_find_max(Af, M, H_s, vep, th0)

th = th0;  %given inital th

lr = 1e-6;

N = max(size(Af));
A_ini = eye(N);
% v = ones(N-1,1);
% H = diag(v,1) + diag(v,-1);
% H = H_s* diag( vep ) *H_s';
while 1
    U_th = H_s* diag( exp(-1i*th*vep) ) *H_s';   %encoding operator theta
    A_th = Af*U_th;
    
    dth = fn_OverLap_pd(A_ini, A_th, M);
    if abs(dth)<1e-7
        break;
    end
    th = th + lr*dth;

end

res = th;

end