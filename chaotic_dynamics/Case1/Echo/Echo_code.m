%% 台球系统 eta=1 m2/m1=1  m1=1
%tan(pi/4)  tan(pi/6) tan(pi/5)  tan(pi*(3-sqrt(5))/4)
clear
tic

%% parameter
th = pi/4;   % billiard angle
high=tan(pi/4);
g = [2, 2, 2;...    % geometry
    0, 1, 1;...
    1, 1, 0;...
    0, 0, high;...
    0, high, 0;...
    1,1,1;0,0,0];
b=[1,1,1;    % boundary
    1,1,1;
    1,1,1;
    1,1,1;
    1,1,1;
    1,1,1;
    48,48,48;
    48,48,48;
    49,49,49;
    48,48,48];

%% mesh
[p,e,t] = initmesh(g,'Hmax',0.1); % create initial mesh
Num_limitation_mesh = 10*10^3;   % number of node
while size(p,2)<Num_limitation_mesh    % 如果生产的节点数没达到要求则继续细化
    [p,e,t]=refinemesh(g,p,e,t);
end
figure;  pdemesh(p,e,t);    % show grid

%% FEM matrix
c=1; a=1; f=0;  % coefficients
[K,M,F,Q,G,H,R]=assempde(b,p,e,t,c,a,f);  % create element matrix
[Kc,Fc,B,ud] = assempde(K,M,F,Q,G,H,R);   % element matrix with boundary
M_bd=B'*M*B;    % eliminates all Dirichlet boundary conditions
K_bd=B'*K*B;

fprintf('matrix size is : %d \n',size(K_bd));

%% initial value
x_centr = 2*1/3;  y_centr = high/3;   % the central point of billiard
initfunc = @(x,y) exp(200*(-(x-x_centr)^2-(y-y_centr)^2));  % Guass wavepacket

u0 = zeros(size(p,2),1);
for i = 1:size(p,2)
    x = p(1,i);
    y = p(2,i);
    value = initfunc(x,y);
    u0(i,1) = value;
end
boundary=[];
for i = 1:size(e,2)
    ind1 = e(1,i);
    ind2 = e(2,i);
    u0(ind1,1) = 0;
    u0(ind2,1) = 0;
    boundary=[boundary; ind1];
end
u_init = u0;
u_init(boundary,:)=[];  % inital value

norm_factor = u0' * M * u0   % normalization factor, = u_init' * M_bd * u_init
figure, pdesurf(p,t,abs(u0(:,1)).^2/norm_factor)   % plot wavepacket probability distribution

clearvars -except K_bd M_bd u_init th boundary p t norm_factor

%% solving ODE with ode45 function
L = 10000; T = 3;
tspan = linspace(0,T,L+1);  

options = odeset('Mass',2*M_bd);    %质量矩阵
options=odeset(options,'RelTol',1e-5);
options=odeset(options,'AbsTol',1e-8);
options=odeset(options,'Stats','on');
options=odeset(options,'JConstant','on');

options=odeset(options,'Jacobian',-1i*K_bd);

[T1,U1]=ode45(@(T1,U1) -1i*K_bd*U1,tspan,u_init,options);  % K_bd*U=i*2*M_bd*dU/dt
U1 = transpose(U1);  % U1(:,t1)表示t1时刻的解


%% Loschmidt echo
echo = zeros(max(size(T1)),1);
for ii=1:max(size(T1))
    amplitude = U1(:,ii)' * M_bd * u_init /norm_factor;
    echo(ii) = abs(amplitude)^2;     % echo = |<phi_t|phi_0>|^2
end

endtime = toc()
save('echo_no1.mat','T1','echo','U1','-v7.3');

%% plot echo at every time
figure
plot(T1,echo,'bo-','linewidth',4)
xlabel('$time$','interpreter','latex','fontsize',35)
ylabel('$Echo$','interpreter','latex','fontsize',35)

%% plot wave function at every time
figure(99)
wave_func = zeros(size(p,2), 1);
for k=1:max(size(T1))
    % 填充边界值，之前把边界部分已删去
    j=1;
    for i=1:size(p,2)
        if ismember(i,boundary) == 1   %判断数组元素i是否属于boundary
            wave_func(i) = 0;
        else
            wave_func(i) = U1(j,k);
            j = j+1;
        end
    end
    pdesurf(p,t, abs(wave_func).^2/norm_factor)
    axis([0 1 0 1 0 300])
    title('time=',T1(k))
    pause(0.5)
end
