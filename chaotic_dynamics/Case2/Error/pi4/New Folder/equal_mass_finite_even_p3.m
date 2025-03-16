%% Calculating Energy of finite interaction in even parity
clear
tic                                    
load('equal_BA_initial.mat','root_ba_e')

%% 定义参数
num=2500;        %要计算的根的数量
root_inf=root_ba_e;
c_max=10;         %你需要到达的最小值c  
c_min=1;

%%
Bethe_roots=zeros(num,2);   %存根
start_num=1;                %=1
end_num=start_num+num-1;    %=num
for jj=start_num:1:end_num   %循环1 ---对根的组数循环,把需要迭代的根先拿出来
    
    x0(1,1)=root_inf(jj,1);    %取根作为初值   %$$$$$$$$$$$$$$$$$$$$$
    x0(2,1)=root_inf(jj,2);                    %$$$$$$$$$$$$$$$$$$$$$

%     Value(jj,1)=norm(fun_BAE_e_initial(c_min,x0));  %验证c=0初值BA根  good 
%     if Value(jj)>0.001
%        fprintf('Initial value is too far--root jj=%g,c=%g, ------\n',jj,c_min)
%        pause
%     end
  
% %{
    %用上一个相互作用的解迭代下一个相互作用的解
    c=0;
    while c+0.0001<c_max         %循环2 ---对不同相互作用循环，
        %根据相互作用范围大小来设置步长
        if c < 10^(-3)
           c_step=10^(-4);     %设置c取值区间的步长          %%%%%%%%%%%%%%%%%
        elseif c<10^(-2)
                c_step=10^(-3);  
        elseif c<10^(-1)
                c_step=10^(-2);
        elseif c<1
                c_step=0.01;
        elseif c<10
            c_step=0.1;
        elseif c<20
            c_step=1;
        end
        
        c=c+c_step;
        
        
%---------------判断初值是否靠近0------------
%要求0.001*c_max*3/2 > 2*( abs(sqrt(3)*root_inf(num,1))+abs(3*root_inf(num,2)))/2
        if norm(fun_BAE_e_initial(c,x0))>0.001 
          %如果初值给得不够靠近方程的根(相差>1)则发出警告: 第几(j)个初值根有问题
           fun_BAE_e_initial(c,x0)
           fprintf('Initial value is too far--root jj=%g,c=%g, ------\n',jj,c)
           c=0;
           pause
            
        else   
%---------------用fsolve代替迭代，似乎可行？暂时没问题------------
        opt=optimset('Display','off','TolFun',1e-16,'TolX',1e-15,...
            'MaxFunEvals',15000,'MaxIter',2000);
        x1=fsolve(@(lam)fun_BAE_e_initial(c,lam),x0,opt);
        x0=x1;
           if fun_BAE_e_initial(c,x0)>10^(-6)
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
% save('equal_BA_finite_even_c1.mat')                        
