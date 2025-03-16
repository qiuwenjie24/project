%% overlap of Gussian state for itself <A|B>
%state matrix-A,B particle number-M 
function res = fn_OverLap(A,B,M)

[Q,R] = qr(A);
C = Q'*B;   %inv(Q)*B; 

% tepA = det(R(1:M,1:M));
tep1 = diag(R);
tepA = prod( tep1(1:M) ,'all'); 
tepB = det( C( 1:M, 1:M) );
res = conj(tepA)*tepB;


end
