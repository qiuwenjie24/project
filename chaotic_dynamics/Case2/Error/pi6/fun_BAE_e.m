function G=fun_BAE_e(c,lam)          %BA方程 even parity
% d=c/2;
u=lam';       %u(1)=k_x  u(2)=k_y
G=zeros(2,1); %2行1列的数组，
 
%% even parity
F_1= c*sin( (u(1)-sqrt(3)*u(2))/2 )- ( u(1)-sqrt(3)*u(2) )*...
    ( cos( (u(1)+u(2)/sqrt(3))/2 )- cos( (u(1)-sqrt(3)*u(2))/2 ) );

F_2= c*sin( (u(1)+sqrt(3)*u(2))/2 )- ( u(1)+sqrt(3)*u(2) )*...
    ( cos( (u(1)-u(2)/sqrt(3))/2 )- cos( (u(1)+sqrt(3)*u(2))/2 ) ); 

F_3= c*sin( u(1) )- 2*u(1)*( cos( u(2)/sqrt(3) )- cos( u(1) ) );


%% 函数列矢量G
% even
G(1,1)=F_1/c;
G(2,1)=F_2/c;
G(3,1)=F_3/c;

end