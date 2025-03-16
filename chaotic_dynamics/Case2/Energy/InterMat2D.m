function N = InterMat2D(p,diag,line,n_ele)

np = size(p,2);
nli = size(line,2);
N = sparse(np,np);  %sparse   zeros
for K = 1:nli
loc2glb = line(1:2,K); % local-to-global map
x = diag(1,K:(K+1)); % node x-coordinates
y = diag(2,K:(K+1)); % node y-
%------------------------
gh1=x(2)-x(1); gh2=y(2)-y(1);
length=norm(gh1*gh2);
NK = [2 1; 1 2]/6 * length * n_ele; % element interaction matrix 
%-------------------------
N(loc2glb,loc2glb) = N(loc2glb,loc2glb)+ NK; % add element interact to M
%----------------------------
end

