%% find th of maximum probability by SDM
%state matrix-A=Af*U_th, particle number-M 
%some posible value th - vaule_th
function res = fn_find_max_old0(Af, M, H_s, vep, vaule_th)

num_th = max(size(vaule_th));
OverLap_th = zeros(1,num_th);
for jj1 = 1:num_th
th = vaule_th(jj1);  %given inital th

U_th = H_s* diag( exp(-1i*th*vep) ) *H_s';   %encoding operator theta
A_th = Af*U_th; 

OverLap = fn_OverLap(A_th,A_th,M);
OverLap_th(jj1) = OverLap;
end

OverLap_th = real(OverLap_th);
[~, po_th] = max(OverLap_th);
res = vaule_th(po_th);

end