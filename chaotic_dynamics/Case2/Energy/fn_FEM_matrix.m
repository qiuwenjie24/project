%% 有限元矩阵 A, N, M
%通过FEM将定态薛定谔方程变为稀疏矩阵本征值问题
% x in [0,sqrt(m1)*L], y in [0,sqrt(m2)*L]


function [A, N, M, A_odd,N_odd,M_odd, A_even,N_even,M_even, p,t]...
         = fn_FEM_matrix(m1,m2)
%% 参数
% tic
L=1;
mh1=sqrt(m1)*L; mh2=sqrt(m2)*L;    %总矩形网格的长mh1，和高mh2
%划分网格较细时创建矩阵很大,占内存很大运算很慢  0.001精度至少要300  700--3h
n_ele=700;     
n_nod=n_ele+1;    %假设有一行(列)有n_ele个单元,有n_nod个节点
len=mh1/n_ele; hig=mh2/n_ele; %单位长和高
n_p=n_nod*n_nod; n_t=n_ele*n_ele*2;   %总节点数，总单元数

%编号
% (h-1)h+1, (h-1)h+2,  ...,   h*h
%     ...      ...     ...    ...
%    2h+1,     ...,    ...,   3h
%     h+1,     h+2,    ...,   2h
%       1,       2,    ...,    h
%% 获取p矩阵
p=zeros(2,n_p); t=zeros(3,n_t);    %创建p,t矩阵
for i=1:1:n_p       %以1节点为原点
    a2=mod(i,n_nod);    %余数{1,2,...,(n_nod-1),0}
    if a2==0            %修改成{1,2,...,(n_nod-1),n_nod}
        a2=n_nod;
    end
    a1=(i-a2)/n_nod + 1;    %该节点位于第a1行第a2列
    p(1,i)=(a2-1)*len;   %节点坐标
    p(2,i)=(a1-1)*hig;
end
% end_time1=toc()  %1
%% 获取t矩阵
for i=1:1:n_t
    b2=mod(i,n_ele);      %余数{1,2,...,(n_ele-1),0}
    if b2==0              %修改成{1,2,...,(n_ele-1),n_ele}
        b2=n_ele;
    end
    b1=(i-b2)/n_ele + 1;    %该单元位于第b1(单元)行第b2(单元)列
    b3=mod(b1,2);
    t(4,i)=1;               %第四行貌似删除了也不影响？
    
    if b3==1      %奇数       
    c1=(b1+1)/2;        %b1是单元行，c1是点行
    t(1,i)=((c1-1)*n_nod)+b2;
    t(2,i)=t(1,i)+1;
    t(3,i)=t(2,i)+n_nod; 
    
    elseif b3==0  %偶数    
    c1=b1/2;           %逆时针排序
    t(1,i)=((c1-1)*n_nod)+b2;
    t(2,i)=t(1,i)+1 + n_nod;
    t(3,i)=t(1,i)+n_nod; 
    else
        fprintf('---wrong---')
    end
    
end
% end_time2=toc()   %2
%% 获取对角线节点矩阵 diag line      
n_d=n_nod;    %对角线的节点
n_e=n_ele;    %对角线的线段单元
diag=zeros(2,n_d);     %对角线上节点坐标
line=zeros(2,n_e);    %对角线上边单元，即单元线段的两个节点
for i=1:1:n_d
    diag(1,i)=(i-1)*len;
    diag(2,i)=(i-1)*hig;
end
for i=1:1:n_e
    line(1,i)=(i-1)*n_nod+i;          %该节点所处位置为i行i列
    line(2,i)=line(1,i)+1+n_nod;      %该节点所处位置为(i+1)行(i+1)列
end
% end_time3=toc()   %3
%% 画网格  
%这里是手动划分，自己设定p,t矩阵
% [p,e,t] = initmesh('lshapeg','hmax',0.999); % create 'lshapeg' mesh
%需要弄清楚e矩阵，然后求出e矩阵，再显示网格
%pdemesh(p,e,t)    %显示网格
%% 组装矩阵
M = MassMat2D(p,t);
A = StiffMat2D(p,t);
N = InterMat2D(p,diag,line,n_ele);

% end_time4=toc()    %4
%% 删除边界节点的贡献的行列
%因为边界条件，边界节点对无贡献
bottom=(1:n_nod);
top=((n_nod-1)*n_nod+1:n_nod*n_nod);
left=((n_nod+1):n_nod:((n_nod-2)*n_nod+1));
right=(2*n_nod:n_nod:(n_nod-1)*n_nod);
boundary=[bottom top left right];  %储存边界节点
tot=size(bottom,2)+size(top,2)+size(left,2)+size(right,2);   %边界上总节点数

 M(boundary,:)=[];     %删除M A N矩阵的第j行第j列
 M(:,boundary)=[];
 A(boundary,:)=[];
 A(:,boundary)=[];
 N(boundary,:)=[];
 N(:,boundary)=[];
 
% end_time5=toc()    %5
% fprintf('matrix size is : %d \n',size(A,1));


%% even parity 
num_max = size(A,1);
num_mid = round(size(A,1)/2);
M_even = M(1:num_mid, 1:num_mid) + fliplr(M(1:num_mid, num_max+1-num_mid:num_max))...
       + flipud(M(num_max+1-num_mid:num_max, 1:num_mid))...
       + rot90(M(num_max+1-num_mid: num_max, num_max+1-num_mid: num_max),2);

A_even = A(1:num_mid, 1:num_mid) + fliplr(A(1:num_mid, num_max+1-num_mid:num_max))...
       + flipud(A(num_max+1-num_mid:num_max, 1:num_mid))...
       + rot90(A(num_max+1-num_mid: num_max, num_max+1-num_mid: num_max),2);

N_even = N(1:num_mid, 1:num_mid) + fliplr(N(1:num_mid, num_max+1-num_mid:num_max))...
       + flipud(N(num_max+1-num_mid:num_max, 1:num_mid))...
       + rot90(N(num_max+1-num_mid: num_max, num_max+1-num_mid: num_max),2);
   
%% odd parity
M_odd = M(1:num_mid, 1:num_mid) - fliplr(M(1:num_mid, num_max+1-num_mid:num_max))...
       - flipud(M(num_max+1-num_mid:num_max, 1:num_mid))...
       + rot90(M(num_max+1-num_mid: num_max, num_max+1-num_mid: num_max),2);

A_odd = A(1:num_mid, 1:num_mid) - fliplr(A(1:num_mid, num_max+1-num_mid:num_max))...
       - flipud(A(num_max+1-num_mid:num_max, 1:num_mid))...
       + rot90(A(num_max+1-num_mid: num_max, num_max+1-num_mid: num_max),2);

N_odd = N(1:num_mid, 1:num_mid) - fliplr(N(1:num_mid, num_max+1-num_mid:num_max))...
       - flipud(N(num_max+1-num_mid:num_max, 1:num_mid))...
       + rot90(N(num_max+1-num_mid: num_max, num_max+1-num_mid: num_max),2);

if mod(num_max,2) == 1
   A_odd(num_mid,:) = []; A_odd(:,num_mid) = [];
   M_odd(num_mid,:) = []; M_odd(:,num_mid) = [];
   N_odd(num_mid,:) = []; N_odd(:,num_mid) = [];
end

end



