%% Calculating Energy of finite interaction in even parity
clear
tic                                    
load('unequal_BA_inf.mat','root_ba')
% load('BAroots_finite_even_100.mat', 'Bethe_roots')
% root_inf=Bethe_roots;
%% �������
num=2500;        %Ҫ����ĸ�������
root_inf=root_ba(1:num,:);
c_max=5*10^6;      %�໥����c�����ֵ��Ҫ���ƣ�����ѡ��d�Ľ��������Ľ��Զ
c_min=100;         %����Ҫ�������Сֵc  

%�����ֵ��������BA���̵����ƫ��ֵΪ��2*( abs(sqrt(3)*kx) + abs(3*ky) )/2/d_max
%����Ҫ���ֵnorm(fun_BAE(d,x0))< ���ƫ��ֵ����������d_maxʹ�����ƫ��ֵ������С
%Ҫ��0.001*c_max*3/2 > 2*( abs(sqrt(3)*root_inf(num,1))+abs(3*root_inf(num,2)))/2

%���ֵMA 10    | 100  | 1000  | 10^4  | 10^5  | inf
%��СֵMI 1     | 10   | 100   | 1000  | 10^4  | 10^5
%����ST   0.001 | 0.01 | 0.1   | 1     | 10    | 100
%5W����           0.001  0.01   0.1     1
%         5       4      3      2       1    

%%
Bethe_roots=zeros(num,2);   %���
start_num=1;                %=1
end_num=start_num+num-1;    %=num
for jj=start_num:1:end_num   %ѭ��1 ---�Ը�������ѭ��,����Ҫ�����ĸ����ó���
    
    x0(1,1)=root_inf(jj,1);    %ȡ����Ϊ��ֵ   %$$$$$$$$$$$$$$$$$$$$$
    x0(2,1)=root_inf(jj,2);                    %$$$$$$$$$$$$$$$$$$$$$

%     Value(jj,1)=norm(fun_BAE_e(c_max,x0));  %��֤�����ֵBA��  good  2e-04
%     if Value(jj)>0.001
%        fprintf('Initial value is too far--root jj=%g,c=%g, ------\n',jj,c_max)
%        pause
%     end
  
% %{
    %����һ���໥���õĽ������һ���໥���õĽ�
    c=c_max;
    while c>c_min         %ѭ��2 ---�Բ�ͬ�໥����ѭ����
        %�����໥���÷�Χ��С�����ò���
        if c > 10^5
           c_step=100;     %����cȡֵ����Ĳ���          %%%%%%%%%%%%%%%%%
        elseif c>10^4
                c_step=50;  
        elseif c>5000
                c_step=1;
        elseif c>1000
                c_step=0.5;
        elseif c>100
                c_step=0.05;
        elseif c>10
                c_step=0.01;
        else
            c_step=0.001;
        end
        
        c=c-c_step;

%---------------�жϳ�ֵ�Ƿ񿿽�0------------
%Ҫ��0.001*c_max*3/2 > 2*( abs(sqrt(3)*root_inf(num,1))+abs(3*root_inf(num,2)))/2
        if norm(fun_BAE_e(c,x0))>0.001 
          %�����ֵ���ò����������̵ĸ�(���>1)�򷢳�����: �ڼ�(j)����ֵ��������
           fun_BAE_e(c,x0)
           fprintf('Initial value is too far--root jj=%g,c=%g, ------\n',jj,c)
           c=0;
           pause
            
        else   
%---------------��fsolve����������ƺ����У���ʱû����------------
        opt=optimset('Display','off','TolFun',1e-16,'TolX',1e-15,...
            'MaxFunEvals',15000,'MaxIter',2000,'Algorithm','Levenberg-Marquardt');
        x1=fsolve(@(lam)fun_BAE_e(c,lam),x0,opt);
        x0=x1;
           if fun_BAE_e(c,x0)>10^(-6)
              %fsolve���Ľⲻ���򷢳�����
              pfrintf('fsolve has wrong at c=%d,j=%d \n',c,jj)
              pause
           end
           
        end


    end      %ѭ��2 ------�Բ�ͬ�໥����ѭ��
%  %}     
    Bethe_roots(jj,:)=x0';
    jj
    %�õ�ָ���໥�����µķ��̽⣬�����뵽Bethe_roots�д���    
    
end       %ѭ��1 ---�Ը�������ѭ��
toc

%% ============��������==========

for n=1:num
    
Bethe_roots(n,3)=( Bethe_roots(n,1) )^2 /2+ ( Bethe_roots(n,2) )^2 /2;

end
En_even2=Bethe_roots(:,3);
%=================================
end_time=toc   %
save('BAroots_finite_even_c100.mat')                       
