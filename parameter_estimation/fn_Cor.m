% <At|a+(n) a(n)|At>
%现在At，总长度N, 粒子数M
function res = fn_Cor(At, M)

[Q,R] = qr(At);
tep1 = diag(R);
tepA = prod( tep1(1:M) ,'all'); 
Q_new = Q(:,1:M);

tep1 = transpose(Q_new * Q_new');
res = tep1.*conj(tepA)*tepA;  
res = transpose(res);         %N*N

end


