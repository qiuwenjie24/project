function c_n=fn_c(n,time,fn_E,fn_x,m_cut,k_cut)
c_n=0; 
for m=1:m_cut  % truncation
    ww_nm=fn_b(n,m,time,fn_E,fn_x,k_cut);
    c_n=c_n+ww_nm.*conj(ww_nm);
end
end