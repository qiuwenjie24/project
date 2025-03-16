%% 两体等质量的BA方程 fsolve(不够好)
function main()
tic                                    
load('equal_BA_infinite.mat','root_ba')
AAA=root_ba;
num=108;   
Bethe_roots=zeros(num,2);%------存根
start_num=1;                %=1   
end_num=start_num+num-1;    %=num
% c=100;     % 0.4往下就很难迭代了
% n1=1; n2=2;
for jj=start_num:1:end_num 

k1=AAA(jj,1);k2=AAA(jj,2);
E(jj,4)=(k1^2+k2^2)/2;  

%--------------------fsolve----------------
 n1=k1/pi;   n2=k2/pi;
 k0(1)=n1*pi; k0(2)=n2*pi;
%  fun = @(k) fun_BA(n1,n2,k0,d);
 for d=10000:-10:10 
     if fun_BA(n1,n2,k0,d)>10^(-3)   % 保证初值小
         fprintf('wrong d=%d,jj=%d',d,jj)
         pause
     end

 opt=optimset('Display','off','TolFun',1e-16,'TolX',1e-15,'MaxFunEvals',15000,...
     'MaxIter',2000);
 k_new=fsolve(@(k) fun_BA(n1,n2,k,d),k0,opt);
%  k0-k_new
 k0=k_new;
 fun_BA(n1,n2,k0,d)
 end
 k1=k0(1); 
 k2=k0(2); 
%-------------------------------------------
%--------------------迭代--------------------
% flag=1;
% ii=1;
% n=0;
% while ii
%     
%  kx=AAA(jj,1) - ( fun_theta(k1-k2,c) + fun_theta(k1+k2,c) )/2;
%  ky=AAA(jj,2) - ( fun_theta(k2-k1,c) + fun_theta(k1+k2,c) )/2;
%  k1=kx; k2=ky;
%  n=n+1;
% 
%  f1=AAA(jj,1) - ( fun_theta(k1-k2,c) + fun_theta(k1+k2,c) )/2-kx;
%  f2=AAA(jj,2) - ( fun_theta(k2-k1,c) + fun_theta(k1+k2,c) )/2-ky; 
%  flag=norm(f1^2+f2^2);
%  
%   if flag<10^(-6)
%      ii=0;
%   end
%  if n>100000
%      fprintf('wrong n=%d\n',n)
%      pause
%  end
% end
%-----------------------------------------------------

 E(jj,1)=k1; 
 E(jj,2)=k2; 
 E(jj,3)=(k1^2+k2^2)/2;   
 
end
E(:,1:3)-root_ba_init(1:num,:);
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
