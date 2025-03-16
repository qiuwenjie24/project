%% no2 - integrable 2    %调节温度和时间
clear 
tic
load('Matrix_M_A_N_no2.mat')
c = 0
KK = A+2.*c.*N;
[rm,rv]=eigs(KK,2*M,500,'SM');
endtime=toc()   
save('State_all_no2_c1.mat','m1','m2','c','rv','rm','endtime','-v7.3')
clearvars -except M m1 m2 c rv rm

%%  eigenvalue - fn_E and eigenstate - eign_st_e
eign_st_e=zeros(size(rm,1),500);
fn_E(:,1)=diag(rv);
for ii=1:500
    nol=rm(:,ii)'*M*rm(:,ii);    % normalization
    eign_st_e(:,ii)=rm(:,ii)/sqrt(nol); 
end
% eign_st_e(:,1)'*M*eign_st_e(:,1)=1

toc

error=exp(-1*fn_E(200)/400) 
% error<0.001 则表明用n=350以上本征态，T=400以下温度的thermal OTOC计算结果可接受的

%% fn_x
% [x_Q_all, x_Q_odd, x_Q_even, ~, ~] = fn_FEM_x_Q(m1,m2);
load('matrix_OTOC_all_no2_c5.mat','x_Q_all')

fn_x=zeros(500);
fn_p=zeros(500);  % 用xp的对易关系验证fn_x,fn_p没问题
for n=1:500
    for m=1:500
        fn_x(n,m)=eign_st_e(:,n)'*x_Q_all*eign_st_e(:,m);
        fn_p(n,m)=1i*(fn_E(n)-fn_E(m))*fn_x(n,m);
    end
end
% ss=fn_x*fn_p-fn_p*fn_x-1i*eye(500);  % 用xp的对易关系验证fn_x,fn_p没问题
% ssr=real(ss);       % =0  只需看矩阵左上角部分是否接近0，越远本征态误差增大，
% ssi=imag(ss);       % =0  因为原x,p矩阵是无穷矩阵
% sum_ssr=sum(ssr);   
% ssi(abs(ssi)<10^(-5))=0;
end_time0=toc()
save('matrix_OTOC_all_no2_c1.mat','-v7.3')

time=0:0.001:3;
m_cut=500; k_cut=500;   % truncation
%% microcanonical OTOC   
% 确保取前500本征态计算时n=350时的microcanonical OTOC是准确的
% c_n=fn_c(n,time,fn_E,fn_x,m_cut,k_cut);
c_n_350=fn_c(200,time,fn_E,fn_x,350,350);
toc
c_n_400=fn_c(200,time,fn_E,fn_x,400,400);
toc
c_n_450=fn_c(200,time,fn_E,fn_x,450,450);
toc
c_n_500=fn_c(200,time,fn_E,fn_x,500,500);
end_time1=toc()    
save('mic_OTOC_all_no2_c1.mat',...
    'c_n_350','c_n_400','c_n_450','c_n_500','end_time1','error','c','m1','m2')

%% thermal OTOC
% C_T=fn_Cc(Tem,time,fn_E,fn_x,m_cut,k_cut);
C_T_1=fn_Cc(100,time,fn_E,fn_x,m_cut,k_cut);
toc
C_T_2=fn_Cc(200,time,fn_E,fn_x,m_cut,k_cut);
toc
C_T_3=fn_Cc(300,time,fn_E,fn_x,m_cut,k_cut);
toc
C_T_4=fn_Cc(400,time,fn_E,fn_x,m_cut,k_cut);
end_time2=toc()  
save('ther_OTOC_all_no2_c1.mat',...
     'C_T_1','C_T_2','C_T_3','C_T_4','end_time2','c','m1','m2')   

