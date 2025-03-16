%% measurement probability
%state matrix-A, measurement strength-lambda, particle number-M 
function res = fn_P(A, lambda, M)

[N, ~] = size(A);  %length

tep1 = fn_Cor(A,M);  %N*N
tep2 = transpose( diag(tep1));  %1*N
Cor_ada = tep2./fn_OverLap(A,A,M);
% Cor_ada = Cor_ada.*M./sum(Cor_ada);
% sum(Cor_ada)

vP = (1 + (exp(-lambda)-1)* Cor_ada)/(N + (exp(-lambda)-1)*M);
vP = real(vP);  %probability sum(vP)=1

if abs(sum(vP)-1)>5*1e-8
   fprintf('Error of vP is %d \n',abs(sum(vP)-1));
%    pause(10)
end

vP = vP/sum(vP);
res = vP;
end