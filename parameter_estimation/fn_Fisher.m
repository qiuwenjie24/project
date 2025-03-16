%% Fisher information
%probability-vP, partial derivative of probability-vP
function res = fn_Fisher(vP, vP_pd)

vP_pd = vP_pd(abs(vP)>1e-10);
vP = vP(abs(vP)>1e-10);
res = sum(vP_pd.^2./vP);

end