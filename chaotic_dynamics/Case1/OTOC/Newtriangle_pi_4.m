%通过FEM将定态薛定谔方程变为稀疏矩阵本征值问题
%tan(pi/6)  tan(pi/4)  tan(pi/5)  cot(pi*(sqrt(5)-1)/4)  
clear; 
tic

%% parameter
th = pi/4;   % billiard angle
high=tan(pi/4);
g = [2, 2, 2;...    % geometry
     0, 1, 1;...
     1, 1, 0;...
     0, 0, high;...
     0, high, 0;...
     1, 1, 1;
     0, 0, 0];  
b = [1, 1, 1;    % boundary
     1, 1, 1;
     1, 1, 1;
     1, 1, 1;
     1, 1, 1;
     1, 1, 1;
    48,48,48;
    48,48,48;
    49,49,49;
    48,48,48];

%% mesh
[p,e,t] = initmesh(g,'Hmax',0.1); % create initial mesh
Num_limitation_mesh = 8*10^5;   % number of node
while size(p,2)<Num_limitation_mesh    % 如果生产的节点数没达到要求则继续细化
    [p,e,t]=refinemesh(g,p,e,t);
end 
figure;  pdemesh(p,e,t);    % show grid

%% FEM matrix
c=1; a=1; f=0;  % coefficients
% [K1,M1,F1] = assema(p,t,c,a,f);  % M1=M
[K,M,F,Q,G,H,R]=assempde(b,p,e,t,c,1,f); % create element matrix, Elliptic PDE
[Kc,Fc,B,ud] = assempde(K,M,F,Q,G,H,R);  % M_bd+K_bd=Kc
M_bd=B'*M*B;    % eliminates any Dirichlet boundary conditions
K_bd=B'*K*B;

%%  solve eigenmode problem
[eigenmode,eigenvalue]=eigs(K_bd,M_bd,500,'SM');% here mass of particle is m=1; 2E 
rv = diag(eigenvalue) /2;   % E=p^2 /2


%%  Plus boundary node and normalization 
for ii=1:500
    rm(:,ii)=B*eigenmode(:,ii);  % plus boundary node effect
    nol=rm(:,ii)'*M*rm(:,ii);    % normalization
    rm(:,ii)=rm(:,ii)/sqrt(nol);
end
% rm(:,1)'*M*rm(:,1)=1
% eigenmode(:,1)'*M_bd*eigenmode(:,1)=1

endtime = toc   % 
save('Newtriangle_pi_4.mat','p','t','rm','rv','high','endtime','-v7.3')



