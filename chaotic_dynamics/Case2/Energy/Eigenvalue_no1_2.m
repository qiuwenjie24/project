%% no1 - integrable 1
clear 
%% FEM matrix A, N, M
% integrable 1
% [A, N, M, A_odd,N_odd,M_odd, A_even,N_even,M_even, p,t] = fn_FEM_matrix(m1,m2);
load('Matrix_M_A_N_no1.mat')
A_all = A; 
N_all = N; 
M_all = M;
tic
%% eigenvalue equation (A+2c*N) * Xi=2E * M * Xi
% --------------------odd parity----------------------------
c = 0
KK_odd = A_odd+2.*c.*N_odd;
En_odd = eigs(KK_odd,2.*M_odd,2200,'SM');   
save('Energy_odd_no1_c1.mat','m1','m2','c','En_odd','-v7.3')
toc

c = 1
KK_odd = A_odd+2.*c.*N_odd;
En_odd = eigs(KK_odd,2.*M_odd,2200,'SM');   
save('Energy_odd_no1_c2.mat','m1','m2','c','En_odd','-v7.3')
toc

c = 10^1
KK_odd = A_odd+2.*c.*N_odd;
En_odd = eigs(KK_odd,2.*M_odd,2200,'SM');   
save('Energy_odd_no1_c3.mat','m1','m2','c','En_odd','-v7.3')
toc

c = 10^2
KK_odd = A_odd+2.*c.*N_odd;
En_odd = eigs(KK_odd,2.*M_odd,2200,'SM');   
save('Energy_odd_no1_c4.mat','m1','m2','c','En_odd','-v7.3')
toc

c = 10^12
KK_odd = A_odd+2.*c.*N_odd;
En_odd = eigs(KK_odd,2.*M_odd,2200,'SM');   
save('Energy_odd_no1_c5.mat','m1','m2','c','En_odd','-v7.3')
toc

% --------------------all parity----------------------------
c = 0
KK_all = A_all+2.*c.*N_all;
En_all = eigs(KK_all,2.*M_all,2200,'SM');   
save('Energy_all_no1_c1.mat','m1','m2','c','En_all','-v7.3')
toc

c = 1
KK_all = A_all+2.*c.*N_all;
En_all = eigs(KK_all,2.*M_all,2200,'SM');   
save('Energy_all_no1_c2.mat','m1','m2','c','En_all','-v7.3')
toc

c = 10^1
KK_all = A_all+2.*c.*N_all;
En_all = eigs(KK_all,2.*M_all,2200,'SM');   
save('Energy_all_no1_c3.mat','m1','m2','c','En_all','-v7.3')
toc

c = 10^2
KK_all = A_all+2.*c.*N_all;
En_all = eigs(KK_all,2.*M_all,2200,'SM');   
save('Energy_all_no1_c4.mat','m1','m2','c','En_all','-v7.3')
toc

c = 10^12
KK_all = A_all+2.*c.*N_all;
En_all = eigs(KK_all,2.*M_all,2200,'SM');   
save('Energy_all_no1_c5.mat','m1','m2','c','En_all','-v7.3')
toc
