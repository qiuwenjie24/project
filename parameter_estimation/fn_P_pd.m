%% partial derivative of measurement probability for theta
%state matrix-A, measurement strength-lambda, particle number-M 
function res = fn_P_pd(A, lambda, M)

[N, ~] = size(A);  %length

A0 = eye(N);       %matrix of initial state

tep1 = fn_Cor_pd(A0, A, N, M);          %1*N
tep2 = transpose( diag(fn_Cor(A,M)) );  %1*N
tep3 = fn_OverLap_pd(A0, A, M);         %1*1
tep4 = fn_OverLap(A, A, M);

Cor_ada_pd = tep1./tep4 - tep2.*tep3./( (tep4).^2);

vP_pd = ( (exp(-lambda)-1)* Cor_ada_pd )/( N + (exp(-lambda) - 1)*M );
vP_pd = real(vP_pd);
res = vP_pd;

end
