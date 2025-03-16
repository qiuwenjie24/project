%% thermal otoc
function C=fn_Cc(T,time,fn_E,fn_x,m_cut,k_cut)
C_o=0; beta=1/T; Z=0; % Z=tr(exp(-beta*H))
for n=1:500   % truncation
    Z=Z+(exp(-1*beta*fn_E(n)));
    w_n=exp(-1*beta*fn_E(n))*fn_c(n,time,fn_E,fn_x,m_cut,k_cut);
    C_o=C_o+w_n;
end
C=C_o/Z;
end