%% 两体等质量的BA方程 循环迭代nice
function main()
tic                                    
load('equal_BA_inf.mat','root_ba')
root_init = root_ba;

num=110000;   
Bethe_roots=zeros(num,3);%------存根
start_num=1;                %=1   
end_num=start_num+num-1;    %=num
c=10^(12);     %相互作用 

for jj=start_num:1:end_num 
    jj
    k0(1)=root_init(jj,1);
    k0(2)=root_init(jj,2);            %初值
    n1=k0(1)/pi; n2=k0(2)/pi;         %量子数
    
%-----------------------------------------------------------------
 ii=1;
 flag=1;
 n=0;
while ii
    
 k1(1)=n1*pi - ( fun_theta(k0(1)-k0(2),c) + fun_theta(k0(1)+k0(2),c) )/2;
 k1(2)=n2*pi - ( fun_theta(k0(2)-k0(1),c) + fun_theta(k0(1)+k0(2),c) )/2;
 n=n+1;
%  f1=n1*pi - ( fun_theta(k1-k2,c) + fun_theta(k1+k2,c) )/2 -kx
%  f2=n2*pi - ( fun_theta(k2-k1,c) + fun_theta(k1+k2,c) )/2 -ky 
 flag=norm(k0(1)-k1(1)) + norm(k0(2)- k1(2));
 k0(1)=k1(1); k0(2)=k1(2);
 
 if flag<10^(-9)
     ii=0;
 end
 if n>10000000   %c越靠近0,迭代次数越多，目前没发现迭代失败
     fprintf('wrong n=%d\n',n)
     pause
 end
end
%-----------------------------------------------------

 if norm(fun_BA(n1,n2,k0,c))>0.01
     fprintf('wrong n=%d\n',n)
 end
%------------------------------------------------------
 
 E(jj,1)=k0(1); 
 E(jj,2)=k0(2); 
 E(jj,3)=(k0(1).^2+k0(2).^2)/2;   % boson
 E(jj,4)=((n1*pi)^2+(n2*pi)^2)/2;  % fermon
end
E_bose=sort( E(:,3) );
E_fermi=sort( E(:,4) );
E_all=sort( [E(:,3);E(:,4)] );  % boson+fermon
% save('equal_BA_c5.mat','E_bose','E_bose','E_all','E','c')
toc()
end

function y=fun_theta(x,c)
y = 2*atan(x/c);
end

function f=fun_BA(n1,n2,k0,c)
k1=k0(1); k2=k0(2);
f(1) = n1*pi - ( fun_theta(k1-k2,c) + fun_theta(k1+k2,c) )/2 -k1;
f(2) = n2*pi - ( fun_theta(k2-k1,c) + fun_theta(k1+k2,c) )/2 -k2;

end
