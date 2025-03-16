%% random (index) result of measurement
%probability-vp
function n = fn_rand_ind(vp)

vp = vp/sum(vp);
rng('shuffle');  %rng('shuffle')
q = rand();   %random number of (0,1)
n = 0;
while q>0
    n = n + 1;
    q = q - vp(n);
end

end