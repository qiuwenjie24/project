%Energy in the case of infinite interaction strength 
%% 计算c极限时的能量(质量比为1)  m_1=1; m_2=1； c=0 or c=inf
clear 

%% 构建总能量矩阵   |n1|,|n2|<=N
% z=x^2+y^2对应圆，放缩到以2y为边长的正方形，圆的解全部在正方形的解里面。
%圆中的最大能量为E= (y^2)* pi^2 < (N^2)* pi^2 /2 = 1.2337e+08 (N=5000)
%前110000能级(保证没漏能级)，第110000个能量= 1.3880e+06(c=inf, bose or fermi)
%                                            1.3833e+06(c=0，bose)  
%                                            6.9352e+05(c=0，bose+fermi)
N=5000;
x=-N:N;     %行向量[-N,...,N]
X=repmat(x,2*N+1,1);      %[-N,...,N; -N,...,N; ...] (2*N+1)*(2*N+1)矩阵
Y=X';              

k_x=X.*pi;                  %k_x=n1*pi
k_y=Y.*pi;                  %k_y=n2*pi

E_x=k_x.^2;             %E_x=k_x^2 
E_y=k_y.^2;             %E_y=k_y^2  
E_sum=(E_x+E_y)/2;    %E_sum=(E_x + E_y)/2 =(n1^2+n2^2)*pi^2 /2   总能量

%% 筛选符合要求的能量
Bethe_roots=zeros((2*N+1)*(2*N+1),3);
D_1=(2*N+1)*(2*N+1)-1;    %E_sum矩阵中元素个数共有(2*N+1)*(2*N+1)个
for k=0:D_1             
    i=mod(k,2*N+1)+1;   %i={1,2,...,(2*N)+1}表示第i列   共出现(2*N+1)次 
    j=(k-mod(k,2*N+1))/(2*N+1)+1; %j={1,2,...,(2*N)+1}表示第j行  共出现(2*N+1)次
    ss=(i-N-1)+(j-N-1);      %n2=(i-N-1)   n1=(j-N-1)
    
    u_x=k_x(i,j);  u_y=k_y(i,j);   E=E_sum(i,j);
    %相互作用无穷大时，简并度为2 (全同对称和反对称)
%     if k_y(i,j)>0 && k_x(i,j)>0 && k_x(i,j)<k_y(i,j) %c=infinite要求n1,n2满足不等
    if k_y(i,j)>0 && k_x(i,j)>0 && k_x(i,j)<=k_y(i,j)  %c=0且bose的情形
%     if k_y(i,j)>0 && k_x(i,j)>0                      %c=0则bose+fermi    
       Bethe_roots(k+1,:)=[u_x, u_y, E];     %储存动量k1，k2和能量E  
    end
        
end

r_E_sum_0=sortrows(Bethe_roots,3);        %按能级大小排序

%消去矩阵中元素全为0的行,即清除root_ba_1中未放入解的位置
r_E_sum_1=r_E_sum_0;
r_E_sum_1(all(r_E_sum_1==0,2),:) = [];    
r_E_sum_2=r_E_sum_1;      

num = size(r_E_sum_2,1)
E_max = r_E_sum_2(110000,3)
if num<11*10^4 || E_max>1.25e+07 % 1.2337e+08
    fprintf('wrong \n')
    pause
end

root_ba=r_E_sum_2(1:110000,:);  


%% save

% c=inf; m_1=1; m_2=1
% E_bose = sort(root_ba(:,3)); E_fermi = sort(root_ba(:,3)); 
% E_all = sort( [E_bose;E_fermi] ); 
% save('equal_BA_inf.mat','root_ba','E_bose','E_fermi','E_all','c','m_1','m_2')


% c=0; m_1=1; m_2=1  %bose
% E_bose = sort(root_ba(:,3)); 
% save('equal_BA_zero.mat','root_ba','E_bose','c','m_1','m_2') %c=0 bose    

