%Energy in the case of infinite interaction strength 
%% 计算c极限时的能量(质量比为1)  m_1=1; m_2=1； c=0 且要求额外满足 0<k_X<k_Y
clear 

%% 构建总能量矩阵   |n1|,|n2|<=N
% z=x^2+y^2对应圆，放缩到以2y为边长的正方形，圆的解全部在正方形的解里面。
%圆中的最大能量为E= (y^2)* pi^2 < (N^2)* pi^2 /2 = 1.2337e+08 (N=5000)
%前5w能级(保证没漏能级)，第50000个能量= 1.2644e+06  (even)
%                                       1.2598e+06  (odd)

N=5000;
x=1:N;     %行向量[1,...,N]
X=repmat(x,N,1);      %[1,...,N; 1,...,N; ...] (N)*(N)矩阵
Y=X';              

k_x=X.*pi;                  %k_x=n1*pi
k_y=Y.*pi;                  %k_y=n2*pi

E_x=k_x.^2;             %E_x=k_x^2 
E_y=k_y.^2;             %E_y=k_y^2  
E_sum=(E_x+E_y)/2;    %E_sum=(E_x + E_y)/2 =(n1^2+n2^2)*pi^2 /2   总能量

%% 筛选符合要求的能量
Bethe_roots_e=zeros( N*N,3);
Bethe_roots_o=zeros( N*N,3);
%E_sum矩阵中元素个数共有(N)*(N)个
for k=0:N*N -1            
    i=mod(k,N)+1;        %i={1,2,...,N}表示第i列   共出现N次 
    j=(k-mod(k,N))/N +1; %j={1,2,...,N}表示第j行  共出现N次
    ss=i+j;              %n1=i   n2=j
    
    u_x=k_x(i,j);  u_y=k_y(i,j);   E=E_sum(i,j);
    if mod(ss,2)==0 && k_x(i,j)<k_y(i,j)     %c=0且0<k_x<k_y, even情形
       Bethe_roots_e(k+1,:)=[u_x, u_y, E];     %储存动量k1，k2和能量E 
       
    elseif mod(ss,2)==1 && k_x(i,j)<k_y(i,j) %c=0且0<k_x<k_y, odd情形
       Bethe_roots_o(k+1,:)=[u_x, u_y, E]; 
    end
        
end

%% even
r_E_sum_0_e=sortrows(Bethe_roots_e,3);        %按能级大小排序

%消去矩阵中元素全为0的行,即清除root_ba_1中未放入解的位置
r_E_sum_1_e=r_E_sum_0_e;
r_E_sum_1_e(all(r_E_sum_1_e==0,2),:) = [];    
r_E_sum_2_e=r_E_sum_1_e;      

num_e = size(r_E_sum_2_e,1)
E_max_e = r_E_sum_2_e(50000,3)
if num_e<5*10^4 || E_max_e>1.25e+07 % 1.2337e+08
    fprintf('wrong \n')
    pause
end

root_ba_e=r_E_sum_2_e(1:50000,:);  

%% odd
r_E_sum_0_o=sortrows(Bethe_roots_o,3);        %按能级大小排序

%消去矩阵中元素全为0的行,即清除root_ba_1中未放入解的位置
r_E_sum_1_o=r_E_sum_0_o;
r_E_sum_1_o(all(r_E_sum_1_o==0,2),:) = [];    
r_E_sum_2_o=r_E_sum_1_o;      

num_o = size(r_E_sum_2_o,1)
E_max_o = r_E_sum_2_o(50000,3)
if num_o<5*10^4 || E_max_o>1.25e+07 % 1.2337e+08
    fprintf('wrong \n')
    pause
end

root_ba_o=r_E_sum_2_o(1:50000,:);  

%% save
c=0; m_1=1; m_2=1;  
save('equal_BA_initial.mat','root_ba_e','root_ba_o','c','m_1','m_2') 
%c=0 0<k_x<k_y odd and even

