function G=fun_BAE_o(c,lam)          %BA方程 even parity
u=lam';       %u(1)=k_x  u(2)=k_y
G=zeros(2,1); %2行1列的数组，

%% even parity
F_1= c*sin( u(1)-u(2) )+ ( u(1)-u(2) )*...
    ( 1+ cos( u(1)-u(2) ) );

F_2= c*sin( u(1)+u(2) )+ ( u(1)+u(2) )*...
    ( 1+ cos( u(1)+u(2) ) ); 


%% 函数列矢量G
% even
G(1,1)=F_1/c;
G(2,1)=F_2/c;

end