%% echo - integrable 2 -c3
clear
%% 
tic
m1 = 1; 
m2 = 1/3;
% fn_FEM_matrix_echo(m1,m2);
toc
load('matrix_echo_no2.mat')
%% 求初值

% Guass wavepacket
xx1 = 2*mh1/3; yy1 = mh2/3; % 左下角
initfunc = @(x,y) exp(200*(-(x-xx1)^2-(y-yy1)^2)) ...
                 +exp(200*(-(mh1-x-xx1)^2-(mh2-y-yy1)^2));


u0=zeros(size(p,2),1);
for i = 1:size(p,2)
   x = p(1,i);
   y = p(2,i);
   value = initfunc(x,y);
   u0(i,1) = value;
end

for i = 1:size(boundary,2)
    ind1 = boundary(1,i);
    u0(ind1,1) = 0;
end
pf = u0; 
pf(boundary,:)=[];

% j=1;
% for i=1:n_p  
%     if ismember(i,boundary)==1  %判断数组元素i是否为集数组boundary的成员
%         ppf(i,1) = 0;
%     else
%         ppf(i,1) = pf(j,1);
%         j = j+1;
%     end
% end
indent = pf(:,1)' * M * pf(:,1)   % 归一化因子 = u0(:,1)' * M_all * u0(:,1) 
% figure, pdesurf(p,t,abs(u0(:,1)).^2)   % plot projection
toc
clearvars -except A N M pf m1 m2 

%% ode函数求解常微分方程
% time grid
L = 2000; T = 0.6;
tspan = linspace(0,T,L+1);  % 把[0 T]分成L部分

% 方程和相互作用
c1=10;          
KK1=A + 2*c1.*N;          % KK1*U=i*2*M*dU/dt  
% eigenvalue=eigs(KK1,M,20,'SM');

options = odeset('Mass',2*M);    %质量矩阵
options=odeset(options,'RelTol',1e-5);
options=odeset(options,'AbsTol',1e-8);
options=odeset(options,'Stats','on');
options=odeset(options,'JConstant','on');

options1=odeset(options,'Jacobian',-1i*KK1);
x0=pf;   % inital value condition 

[T1,U1]=ode23(@(T1,U1) -1i*KK1*U1,tspan,x0,options1);
U1 = transpose(U1);  % 列向量的组合
end_time1=toc()

%% 计算两个解函数的overlap随时间的变化
inte=x0' * M * x0     % 归一化因子
  for ii=1:size(T1)
   inte1(ii,1)=U1(:,ii)' * M * U1(:,1)/inte;     %内积
   Int1=abs(inte1(ii,1));                        %模长
   inte_real1(ii)=Int1^2;                     %1-模长平方
  end


end_time3=toc()  
save('data_echo_no2_c3.mat','-v7.3');
% figure, plot(2*T1,inte_real1,'b:o') %,axis([0 TT 0 1.3])
% load('data_test46.mat','T1','inte_real1')