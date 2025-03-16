%Energy in the case of infinite interaction strength 
%% ����c�����ʱ������(������Ϊ1/3)     n1,n2ͬ��ż     m_1=1; m_2=1/3;
clear all   

%% ��������������   |n1|,|n2|<=N
% 3z=3*x^2+y^2��Ӧ��Բ����������2yΪ�߳��������Σ���Բ�Ľ�ȫ���������εĽ����档
% ��Բ�е��������ΪE= (y^2)* pi^2 /2/3 = (N^2)* pi^2 /6 = 4.1123e+07 (N=5000)
% ǰ110000�ܼ�(��֤û©�ܼ�)����1709666������=3.7253e+07
N=5000;
x=-N:N;     %������[-N,...,N]
X=repmat(x,2*N+1,1);      %[-N,...,N; -N,...,N; ...] (2*N+1)*(2*N+1)����    
Y=X';              

k_x=X.*pi;           %k_x=n1*pi
k_y=Y.*pi./sqrt(3);  %k_y=n2*pi/sqrt(3)

E_x=k_x.^2;             
E_y=k_y.^2;             
E_sum=(E_x+E_y)/2;    %E_sum=(E_x + E_y)/2 =(n1^2 + n2^2 /3)*pi^2 /2   ������

%% ɸѡ����Ҫ�������
%ɸѡҪ��n1,n2����ͬ��ż
Bethe_roots=zeros((2*N+1)*(2*N+1),3);
D_1=(2*N+1)*(2*N+1)-1;    %E_sum������Ԫ�ظ�������(2*N+1)*(2*N+1)��
for k=0:D_1      
        
    i=mod(k,2*N+1)+1;   %i={1,2,...,(2*N)+1}��ʾ��i��   ������(2*N+1)�� 
    j=(k-mod(k,2*N+1))/(2*N+1)+1;    %j={1,2,...,(2*N)+1}��ʾ��j��  ������(2*N+1)�� 
    ss=(i-N-1)+(j-N-1);      %n2=(i-N-1)   n1=(j-N-1)
    
    u_x=k_x(i,j); u_y=k_y(i,j); E=E_sum(i,j);
    %��BA��������֪���໥��������ʱ���򲢶�Ϊ2 (����ƺ�ż���),ֻ����һ���ܼ�
    if mod(ss,2)==0 && (i-N-1)>0 && (i-N-1)<(j-N-1)
    %Ҫ��n1,n2����ͬ��ͬż 0< sqrt(3)*u_y < u_x �� 0< n2 < n1
       Bethe_roots(k+1,:)=[u_x, u_y, E];     %���涯��k1��k2������E    
    end
        
end

r_E_sum_0=sortrows(Bethe_roots,3);        %���ܼ���С����

r_E_sum_1=r_E_sum_0;
r_E_sum_1(all(r_E_sum_1==0,2),:) = []; %��ȥ������Ԫ��ȫΪ0����,�����δ������λ��
r_E_sum_2=r_E_sum_1;      

num = size(r_E_sum_2,1)
E_max = r_E_sum_2(110000,3)
if num<11*10^4 || E_max>4.1123e+07
    fprintf('wrong \n')
    pause
end

root_ba=r_E_sum_2(1:110000,:);  

%% save
clearvars -except root_ba
c = inf; m_1 = 1; m_2 = 1/3;
E_even = root_ba(:,3);
E_odd = root_ba(:,3);
E_all = sortrows( [E_even;E_odd] );
% save('unequal_BA_inf.mat')

