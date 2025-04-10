function A = StiffMat2D(p,t)
np = size(p,2);
nt = size(t,2);
A = sparse(np,np);  %sparse  zeros
for K = 1:nt
loc2glb = t(1:3,K); % local-to-global map
x = p(1,loc2glb); % node x-coordinates
y = p(2,loc2glb); % node y-
[area,b,c] = Gradients(x,y);
AK = (b * b' + c * c') * area; % element stiffness matrix
A(loc2glb,loc2glb) = A(loc2glb,loc2glb) + AK; % add element stiffnesses to A
end


