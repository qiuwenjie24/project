%% no4 - chaos
clear 
%% FEM matrix A, N, M
% chaos
tic
f_eta = (3-sqrt(5) )/4;
m1 = 1; 
m2 = (tan(f_eta*pi) )^2;
[A, N, M, A_odd,N_odd,M_odd, A_even,N_even,M_even, p,t] = fn_FEM_matrix(m1,m2);
end_time = toc() 
save('Matrix_M_A_N_no4.mat','-v7.3')

%% eigenvalue equation (A+2c*N) * Xi=2E * M * Xi
c = 0
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no4_c1.mat','m1','m2','c','En_even','-v7.3')
toc

c = 1
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no4_c2.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^1
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no4_c3.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^2
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no4_c4.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^12
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no4_c5.mat','m1','m2','c','En_even','-v7.3')
toc
