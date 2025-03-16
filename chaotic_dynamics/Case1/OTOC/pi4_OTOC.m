%% OTOC    %调节温度和时间
clear
load('Newtriangle_pi_4.mat','rv','rm','p','t','high')

%% matrix of fn_x and fn_p
tic
fn_E=rv;    % 能级
eign_st=rm; % 本征态
error=exp(-1*fn_E(200)/400) 
% error<0.001 则表明n=200以上本征态对thermal OTOC贡献很小，可忽略

fn_x=zeros(500);
fn_p=zeros(500);
x_Q=fn_Q(p,t);  % 有限元基下的坐标矩阵
for n=1:500
    for m=1:500
        fn_x(n,m)=eign_st(:,n)'*x_Q*eign_st(:,m);
        fn_p(n,m)=1i*(fn_E(n)-fn_E(m))*fn_x(n,m);
    end
end
endtime1 = toc()
save('matrix_OTOC_pi4.mat','fn_x','fn_p','endtime1','-v7.3')

%% microcanonical OTOC   
% 确保取前500本征态计算时，n=350的microcanonical OTOC是准确的
% c_n=fn_c(n,time,fn_E,fn_x,m_cut,k_cut);

tic
time=0:0.001:3;
c_n_350=fn_c(200,time,fn_E,fn_x,350,350);
toc
c_n_400=fn_c(200,time,fn_E,fn_x,400,400);
toc
c_n_450=fn_c(200,time,fn_E,fn_x,450,450);
toc
c_n_500=fn_c(200,time,fn_E,fn_x,500,500);
endtime2 = toc()   
save('mic_OTOC_pi4.mat','c_n_350','c_n_400','c_n_450','c_n_500','endtime2','error')

%% thermal OTOC
% C_T=fn_Cc(Tem,time,fn_E,fn_x,m_cut,k_cut);

tic
m_cut=500; k_cut=500;   % truncation
C_T_1=fn_Cc(100,time,fn_E,fn_x,m_cut,k_cut);
toc
C_T_2=fn_Cc(200,time,fn_E,fn_x,m_cut,k_cut);
toc
C_T_3=fn_Cc(300,time,fn_E,fn_x,m_cut,k_cut);
toc
C_T_4=fn_Cc(400,time,fn_E,fn_x,m_cut,k_cut);
end_time3=toc()  
save('ther_OTOC_pi4.mat','C_T_1','C_T_2','C_T_3','C_T_4','endtime3')   


