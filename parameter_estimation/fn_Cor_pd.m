%partial_th <a+(n) a(n)>
%初始A0，现在At，总长度N, 粒子数M
function res = fn_Cor_pd(A0, At, N, M)

B0 = A0;
B0(:,[M,M+1]) = B0(:,[M+1,M]);
Bt = At*B0;

[Q,R] = qr(At);
tep1 = diag(R);   %N*1
tepA = prod( tep1(1:M) ,'all'); 

C = Q'*Bt;
Q_p = Q';

Cor_pd1 = zeros(1,N);
for jj2 = 1:M
    
    for jj1 = [jj2, M+1: N]
        
        if jj1==jj2
            vm = 1:M;
            Cof = 1;
        else
            vm = [1:M,jj1];
            vm(vm==jj2) = [];
            Cof = (-1)^(M-jj2);
        end
        
        for n = 1:N
        tep2 = Q_p(jj1,n)*Q(n,jj2)*Cof*conj(det(C(vm,1:M)))*tepA;
        Cor_pd1(n) = Cor_pd1(n) + tep2;
        end

    end
end

Cor_pd1 = (1j)*Cor_pd1;    %<Bt|a+(n) a(n)|At>
Cor_pd2 = conj(Cor_pd1);   %<At|a+(n) a(n)|Bt>
res = Cor_pd1 + Cor_pd2;

end


