%% no1 - integrable 1
clear
figure
pos3=[0.11,0.11,0.35 0.37];pos4=[0.51,0.11,0.35 0.37];
pos1=[0.11,0.57,0.35 0.37]; pos2=[0.51,0.57,0.35 0.37]; 
%% (a)c=0
subplot('position',pos1); 
clearvars -except pos1 pos2 pos3 pos4
load('Energy_odd_no1_c1.mat')  

E_r=sortrows(En_odd,1) ;    % 排序
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

text(2.5,0.5,'(a) $c=0$','fontsize',30,'interpreter','latex')
set(gca,'fontsize',20,'linewidth',1.5)
% axis([-0.2,5,0,600])

%% (b)c=1
subplot('position',pos2);
clearvars -except pos1 pos2 pos3 pos4 edges
load('Energy_odd_no1_c2.mat')    

E_r=sortrows(En_odd,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 1*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gap
% E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);  
h2.Normalization = 'pdf';   %频数变频率密度

text(2.5,1.0,'(b) $c=1$','fontsize',30,'interpreter','latex')
set(gca,'fontsize',20,'linewidth',1.5)
% axis([-0.2,5,0,600])

%% (c)c=10
subplot('position',pos3); 
clearvars -except pos1 pos2 pos3 pos4 edges
load('Energy_odd_no1_c3.mat')     

E_r=sortrows(En_odd,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.978*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
% E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);  
h2.Normalization = 'pdf';   %频数变频率密度

set(gca,'fontsize',20,'linewidth',1.5)
xlabel('$s/s_{aver}$','interpreter','latex','fontsize',30)
ylabel('Frequency','fontsize',25)

text(2.5,0.8,'(c) $c=10$','fontsize',30,'interpreter','latex')
% axis([-0.2,5,0,600])
%% (d)c=100
subplot('position',pos4); 
clearvars -except pos1 pos2 pos3 pos4 edges
load('Energy_odd_no1_c4.mat')   

E_r=sortrows(En_odd,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.99*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);   
h2.Normalization = 'pdf';   %频数变频率密度

text(2.5,0.65,'(d) $c=100$',...
    'fontsize',30,'interpreter','latex')   
set(gca,'fontsize',20,'linewidth',1.5)
% axis([-0.1,3,0,600])
%% (e)c=10^12
figure
clearvars -except pos1 pos2 pos3 pos4 edges
load('Energy_odd_no1_c5.mat')   

E_r=sortrows(En_odd,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.992*(n-1) );
E_gap_new(jj1:n-1)=[];

s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
% E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);   
h2.Normalization = 'pdf';   %频数变频率密度

text(2.0,0.80,'(e) $c=10^{12}$',...
    'fontsize',30,'interpreter','latex')   
set(gca,'fontsize',20,'linewidth',1.5)
% axis([-0.1,3,0,600])