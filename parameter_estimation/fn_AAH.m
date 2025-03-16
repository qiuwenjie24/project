%% Hamitonian of AA model
%length-L, coupling-Cp, disorder-Dis_str
function [AAH_pe, AAH_op] = fn_AAH(L, Cp, Dis_str)
v = ones(L-1,1);
H_op = diag(v,1) + diag(v,-1);   %open boundary

H_pe = H_op ;
H_pe(1,L) = 1;  H_pe(L,1) = 1;   %periodic boundary

H_op = H_op.*Cp;
H_pe = H_pe.*Cp;

v1 = (1:L);   %randn(1,L);
beta = (sqrt(5)-1 )/2;
Dis_coff = cos(2.*pi.*beta.*v1); 
Dis_term = diag(Dis_coff);       %disorder term

AAH_op = H_op + Dis_str.*Dis_term;
AAH_pe = H_op + Dis_str.*Dis_term;
end