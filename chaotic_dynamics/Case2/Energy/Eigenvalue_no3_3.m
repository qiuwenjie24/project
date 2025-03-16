%% no3 - pesudo-integrable 补充
clear 
%% FEM matrix A, N, M
% pesudo-integrable
tic
load('Matrix_M_A_N_no3.mat')

%% eigenvalue equation (A+2c*N) * Xi=2E * M * Xi
c = 10^3
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no3_c4p3.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^4
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no3_c4p4.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^5
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no3_c4p5.mat','m1','m2','c','En_even','-v7.3')
toc

c = 10^6
KK_even = A_even+2.*c.*N_even;
En_even = eigs(KK_even,2.*M_even,2200,'SM');   
save('Energy_even_no3_c4p6.mat','m1','m2','c','En_even','-v7.3')
toc
