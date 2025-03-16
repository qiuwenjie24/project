clear
figure
pos3=[0.11,0.11,0.35 0.37];pos4=[0.51,0.11,0.35 0.37];
pos1=[0.11,0.57,0.35 0.37]; pos2=[0.51,0.57,0.35 0.37]; 

%% (b)c=1
subplot('position',pos2); 
clearvars -except pos1 pos2 pos3 pos4 edges
load('equal_BA_finite_even_c1.mat','En_even2')
% figure
E_r=sortrows(En_even2,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.99*(n-1) );
E_gap_new(jj1:n-1)=[];
edges = 0:0.1:7 ;  
s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);  
h2.Normalization = 'pdf';   %频数变频率密度

set(gca,'fontsize',20,'linewidth',1.5)
xlabel('$s/s_{aver}$','interpreter','latex','fontsize',30)
ylabel('Frequency','fontsize',25)

text(2.5,0.8,'(c) $b=10$','fontsize',30,'interpreter','latex')
% axis([-0.2,5,0,600])
%% (c)c=10
subplot('position',pos3); 
clearvars -except pos1 pos2 pos3 pos4 edges
load('equal_BA_finite_even_c10.mat','En_even2')
% figure
E_r=sortrows(En_even2,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.99*(n-1) );
E_gap_new(jj1:n-1)=[];
edges = 0:0.1:6 ;  
s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);  
h2.Normalization = 'pdf';   %频数变频率密度

set(gca,'fontsize',20,'linewidth',1.5)
xlabel('$s/s_{aver}$','interpreter','latex','fontsize',30)
ylabel('Frequency','fontsize',25)

text(2.5,0.8,'(c) $c=10$','fontsize',30,'interpreter','latex')
% axis([-0.2,5,0,600])
%% (d)c=100
subplot('position',pos4); 
clearvars -except pos1 pos2 pos3 pos4
load('equal_BA_finite_even_c100.mat','En_even2')
% figure
E_r=sortrows(En_even2,1) ;    % 排序
n=size(E_r,1);
E_gap=zeros(n-1,1);
for i=1:1:n-1
   E_gap(i,1)=E_r(i+1,1)-E_r(i,1);  % 能级间隔
end
E_gap_new = sortrows(E_gap,1);  %排序
jj1 = floor( 0.99*(n-1) );
E_gap_new(jj1:n-1)=[];
edges = 0:0.1:4.5 ;  
s_aver = sum(E_gap_new)/size(E_gap_new,1);    %average energy gasp
E_gap_new/s_aver
h2=histogram( E_gap_new/s_aver,edges);   
h2.Normalization = 'pdf';   %频数变频率密度

text(2.5,0.65,'(d) $c=100$',...
    'fontsize',30,'interpreter','latex')   
set(gca,'fontsize',20,'linewidth',1.5)
% axis([-0.1,3,0,600])