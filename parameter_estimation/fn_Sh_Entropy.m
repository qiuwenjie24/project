%% shanon entropy
%probability-vP
function res = fn_Sh_Entropy(vP)

vP = real(vP);
vP = vP/sum(vP);
vP(vP<1e-10) = [];

res = -sum(vP.*log2(vP));

end