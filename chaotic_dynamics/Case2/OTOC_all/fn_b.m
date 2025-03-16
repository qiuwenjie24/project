function b_nm=fn_b(n,m,time,fn_E,fn_x,k_cut)
b_nm=0; 
for k=1:k_cut  % truncation
    E_km=fn_E(k)-fn_E(m);
    E_nk=fn_E(n)-fn_E(k);
    www_nkm=fn_x(n,k)*fn_x(k,m)*...
        (E_km*exp(1i*E_nk*time)-E_nk*exp(1i*E_km*time));
    b_nm=b_nm+www_nkm;
end
end