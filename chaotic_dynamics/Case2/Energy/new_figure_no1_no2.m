%% no1 - integrable 1 and no2 - integrable 2 even parity
clear
figure
pos3=[0.11,0.11,0.35 0.37];pos4=[0.51,0.11,0.35 0.37];
pos1=[0.11,0.57,0.35 0.37]; pos2=[0.51,0.57,0.35 0.37]; 
%% (a)no1 c=0
subplot('position',pos1); 
clearvars -except pos1 pos2 pos3 pos4
load('.\no1\Energy_even_no1_c1.mat')  

E_r=sortrows(En_even,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 1*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gap
edges = 0:0.1:4 ;       % 区间划分
% E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);    
h2.Normalization = 'pdf';   %频数变频率密度

set(gca,'fontsize',25,'linewidth',1.5)
text(1.5,3.6,'(a) $\eta=1,c=0$','fontsize',35,'interpreter','latex')

%% (b)no1 c=100
subplot('position',pos2); 
clearvars -except pos1 pos2 pos3 pos4 edges
load('.\no1\Energy_even_no1_c4.mat')   

E_r=sortrows(En_even,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.987*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
% E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);   
h2.Normalization = 'pdf';   %频数变频率密度

set(gca,'fontsize',25,'linewidth',1.5)
text(1.5,0.8,'(b) $\eta=1,c=100$', 'fontsize',35,'interpreter','latex') 

%% (c)no2 c=0
subplot('position',pos3); 
clearvars -except pos1 pos2 pos3 pos4
load('.\no2\Energy_even_no2_c1.mat')  

E_r=sortrows(En_even,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.935*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gap
edges = 0:0.1:4 ;       % 区间划分
% E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);    
h2.Normalization = 'pdf';   %频数变频率密度

set(gca,'fontsize',25,'linewidth',1.5)
text(1.5,1.6,'(c) $\eta=\frac{1}{3},c=0$','fontsize',35,'interpreter','latex')
xlabel('$s/s_{aver}$','interpreter','latex','fontsize',30)
ylabel('P(s)','fontsize',30)

%% (d)no2 c=100
subplot('position',pos4); 
clearvars -except pos1 pos2 pos3 pos4 edges
load('.\no2\Energy_even_no2_c4.mat')   

E_r=sortrows(En_even,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.987*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
% E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);   
h2.Normalization = 'pdf';   %频数变频率密度
 
set(gca,'fontsize',25,'linewidth',1.5)
text(1.5,0.65,'(d) $\eta=\frac{1}{3},c=100$','fontsize',35,'interpreter','latex')  
