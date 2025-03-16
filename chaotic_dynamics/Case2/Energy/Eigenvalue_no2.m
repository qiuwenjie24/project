%% no2 - integrable 2
clear 
%% FEM matrix A, N, M
% integrable 2
tic
m1 = 1; 
m2 = 1/3;
[A, N, M, A_odd,N_odd,M_odd, A_even,N_even,M_even, p,t] = fn_FEM_matrix(m1,m2);
end_time = toc() 
save('Matrix_M_A_N_no2.mat','-v7.3')

%% eigenvalue equation (A+2c*N) * Xi=2E * M * Xi
c = 0
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no2_c1.mat','m1','m2','c','En_even','-v7.3')
toc

c = 1
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no2_c2.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^1
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no2_c3.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^2
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no2_c4.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^12
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no2_c5.mat','m1','m2','c','En_even','-v7.3')
toc
