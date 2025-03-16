%% from c=10 to c=1
clear
tic
load('equal_BA_finite_even_c10.mat','Bethe_roots')
root_ba=Bethe_roots;
clear Bethe_roots
%% 定义参数
num=2500;        %要计算的根的数量
root_inf=root_ba;
c_max=10;      
c_min=1;         %你需要到达的最小值c  

%无穷初值带入有限BA方程的最大偏离值为：2*( abs(sqrt(3)*kx) + abs(3*ky) )/2/d_max
%即，要求初值norm(fun_BAE(d,x0))< 最大偏离值，并且设置d_max使得最大偏离值尽可能小
%要求0.001*c_max*3/2 > 2*( abs(sqrt(3)*root_inf(num,1))+abs(3*root_inf(num,2)))/2

%最大值MA 10    | 100  | 1000  | 10^4  | 10^5  | inf
%最小值MI 1     | 10   | 100   | 1000  | 10^4  | 10^5
%步长ST   0.001 | 0.01 | 0.1   | 1     | 10    | 100
%5W以上           0.001  0.01   0.1     1
%         5       4      3      2       1    

%%
Bethe_roots=zeros(num,2);   %存根
start_num=1;                %=1
end_num=start_num+num-1;    %=num
for jj=start_num:1:end_num   %循环1 ---对根的组数循环,把需要迭代的根先拿出来
    
    x0(1,1)=root_inf(jj,1);    %取根作为初值   %$$$$$$$$$$$$$$$$$$$$$
    x0(2,1)=root_inf(jj,2);                    %$$$$$$$$$$$$$$$$$$$$$

%     Value(jj,1)=norm(fun_BAE_e(c_max,x0));  %验证无穷初值BA根  good  2e-04
%     if Value(jj)>0.001
%        fprintf('Initial value is too far--root jj=%g,c=%g, ------\n',jj,c_max)
%        pause
%     end
  
% %{
    %用上一个相互作用的解迭代下一个相互作用的解
    c=c_max;
    while c-10^(-4)>c_min         %循环2 ---对不同相互作用循环，
        %根据相互作用范围大小来设置步长
        if c > 10^5
           c_step=100;     %设置c取值区间的步长          %%%%%%%%%%%%%%%%%
        elseif c>10^4
                c_step=50;  
        elseif c>5000
                c_step=1;
        elseif c>1000
                c_step=0.5;
        elseif c>100
                c_step=0.01;
        elseif c>20
                c_step=0.005;
        elseif c>2
                c_step=0.001;
        else
            c_step=0.0005;
        end
        
        c=c-c_step;

%---------------判断初值是否靠近0------------
%要求0.001*c_max*3/2 > 2*( abs(sqrt(3)*root_inf(num,1))+abs(3*root_inf(num,2)))/2
        if norm(fun_BAE_e(c,x0))>0.001 
          %如果初值给得不够靠近方程的根(相差>1)则发出警告: 第几(j)个初值根有问题
           fun_BAE_e(c,x0)
           fprintf('Initial value is too far--root jj=%g,c=%g, ------\n',jj,c)
           c=0;
           pause
            
        else   
%---------------用fsolve代替迭代，似乎可行？暂时没问题------------
        opt=optimset('Display','off','TolFun',1e-16,'TolX',1e-15,...
            'MaxFunEvals',15000,'MaxIter',2000);
        x1=fsolve(@(lam)fun_BAE_e(c,lam),x0,opt);
        x0=x1;
           if fun_BAE_e(c,x0)>10^(-6)
              %fsolve给的解不好则发出警告
              pfrintf('fsolve has wrong at c=%d,j=%d \n',c,jj)
              pause
           end
           
        end


    end      %循环2 ------对不同相互作用循环
%  %}     
    Bethe_roots(jj,:)=x0';
    jj
    %得到指定相互作用下的方程解，并放入到Bethe_roots中储存    
    
end       %循环1 ---对根的组数循环
toc

%% ============计算能量==========

for n=1:num
    
Bethe_roots(n,3)=( Bethe_roots(n,1) )^2 /2+ ( Bethe_roots(n,2) )^2 /2;

end
En_even2=sort(Bethe_roots(:,3));
%=================================
end_time=toc   %
save('equal_BA_finite_even_c1.mat')                        
