%Energy in the case of infinite interaction strength 
%% 计算c无穷大时的能量(质量比为1/3)     n1,n2同奇偶     m_1=1; m_2=1/3;
clear all   

%% 构建总能量矩阵   |n1|,|n2|<=N
% 3z=3*x^2+y^2对应椭圆，放缩到以2y为边长的正方形，椭圆的解全部在正方形的解里面。
% 椭圆中的最大能量为E= (y^2)* pi^2 /2/3 = (N^2)* pi^2 /6 = 4.1123e+07 (N=5000)
% 前110000能级(保证没漏能级)，第1709666个能量=3.7253e+07
N=5000;
x=-N:N;     %行向量[-N,...,N]
X=repmat(x,2*N+1,1);      %[-N,...,N; -N,...,N; ...] (2*N+1)*(2*N+1)矩阵    
Y=X';              

k_x=X.*pi;           %k_x=n1*pi
k_y=Y.*pi./sqrt(3);  %k_y=n2*pi/sqrt(3)

E_x=k_x.^2;             
E_y=k_y.^2;             
E_sum=(E_x+E_y)/2;    %E_sum=(E_x + E_y)/2 =(n1^2 + n2^2 /3)*pi^2 /2   总能量

%% 筛选符合要求的能量
%筛选要求n1,n2必须同奇偶
Bethe_roots=zeros((2*N+1)*(2*N+1),3);
D_1=(2*N+1)*(2*N+1)-1;    %E_sum矩阵中元素个数共有(2*N+1)*(2*N+1)个
for k=0:D_1      
        
    i=mod(k,2*N+1)+1;   %i={1,2,...,(2*N)+1}表示第i列   共出现(2*N+1)次 
    j=(k-mod(k,2*N+1))/(2*N+1)+1;    %j={1,2,...,(2*N)+1}表示第j行  共出现(2*N+1)次 
    ss=(i-N-1)+(j-N-1);      %n2=(i-N-1)   n1=(j-N-1)
    
    u_x=k_x(i,j); u_y=k_y(i,j); E=E_sum(i,j);
    %由BA波函数可知，相互作用无穷时，简并度为2 (奇宇称和偶宇称),只需求一种能级
    if mod(ss,2)==0 && (i-N-1)>0 && (i-N-1)<(j-N-1)
    %要求n1,n2满足同奇同偶 0< sqrt(3)*u_y < u_x 即 0< n2 < n1
       Bethe_roots(k+1,:)=[u_x, u_y, E];     %储存动量k1，k2和能量E    
    end
        
end

r_E_sum_0=sortrows(Bethe_roots,3);        %按能级大小排序

r_E_sum_1=r_E_sum_0;
r_E_sum_1(all(r_E_sum_1==0,2),:) = []; %消去矩阵中元素全为0的行,即清除未放入解的位置
r_E_sum_2=r_E_sum_1;      

num = size(r_E_sum_2,1)
E_max = r_E_sum_2(110000,3)
if num<11*10^4 || E_max>4.1123e+07
    fprintf('wrong \n')
    pause
end

root_ba=r_E_sum_2(1:110000,:);  

%% save
clearvars -except root_ba
c = inf; m_1 = 1; m_2 = 1/3;
E_even = root_ba(:,3);
E_odd = root_ba(:,3);
E_all = sortrows( [E_even;E_odd] );
% save('unequal_BA_inf.mat')

