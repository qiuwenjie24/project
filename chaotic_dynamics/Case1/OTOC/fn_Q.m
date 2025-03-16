%%
function Qp = fn_Q(p,t)
np = size(p,2); % number of nodes
nt = size(t,2); % number of elements
Qp = sparse(np,np); % allocate x-positon matrix
%sparse与zeros可替换，前者节省内存，后者能更直观显示整个矩阵
for K = 1:nt % loop over elements
    loc2glb = t(1:3,K); % local-to-global map
    x = p(1,loc2glb); % node x-coordinates
    y = p(2,loc2glb); % y
    area = polyarea(x,y); % triangle area
    posi_x=(x(1,1)+x(1,2)+x(1,3))/3;   % average node x-coordinates
    QK = [2 1 1;
          1 2 1;
          1 1 2]/12 * area * posi_x; % element x-positon matrix
    % int(v_i*x*v_j)=posi_x*int(v_i*v_j)=posi_x*MK
    Qp(loc2glb,loc2glb) = Qp(loc2glb,loc2glb) + QK; % add element x-positon to Qp
end
end