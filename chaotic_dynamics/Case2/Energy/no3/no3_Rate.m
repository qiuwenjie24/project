% difference from Wigner distribution for eta
clear

%% parameter

for i1=-1:7
if i1==-1
    load('Energy_even_no3_c1.mat','En_even') 
elseif i1==0
    load('Energy_even_no3_c2.mat','En_even') 
elseif i1==1
    load('Energy_even_no3_c3.mat','En_even') 
elseif i1==2
    load('Energy_even_no3_c4.mat','En_even') 
elseif i1==3
    load('Energy_even_no3_c4p3.mat','En_even') 
elseif i1==4
    load('Energy_even_no3_c4p4.mat','En_even') 
elseif i1==5
    load('Energy_even_no3_c4p5.mat','En_even') 
elseif i1==6
    load('Energy_even_no3_c4p6.mat','En_even') 
elseif i1==7
    load('Energy_even_no3_c5.mat','En_even') 
end
%% level spacing
E_r=sortrows(En_even,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.99*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gap
S_n_new=E_gap_new/s_aver;      %归一化能级间隔

%% difference Rate
S_0=0.47291;   %cross point between possion and wigner fun_P(S_0,be1)=fun_P(S_0,be2)
count=sum(S_n_new<=S_0);
int_P=count/(n-1);
be1=0; be2=1;      %possion for be1=0 and wigner for be2=1
int_P_p=integral(@(x) fun_P(x,be1),0,S_0);
int_P_w=integral(@(x) fun_P(x,be2),0,S_0);
Rate(i1+2)=abs( (int_P- int_P_w)/(int_P_p- int_P_w) )

end
%% Fig.10
figure

plot(10.^(0:6),Rate(1:7),'o-k','LineWidth',2,'MarkerSize',20,...
    'MarkerFaceColor','k')
% axis([0,0.9,0,0.81])
% xlim([0,10^12])
set(gca,'XScale','log')
