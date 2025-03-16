%% Brody function
function P=fn_Brody(s,be)
%% possion be=0;     wigner be=1.
ff=( (be+2)/(be+1) );
b=( gamma(ff) )^(be+1);
P=(be+1).*b.*s.^(be).*exp(-1*b.*s.^(be+1));

end