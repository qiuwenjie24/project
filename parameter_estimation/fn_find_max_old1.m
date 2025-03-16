%% find th of maximum probability by SDM
%state matrix-A=Af*U_th, particle number-M 
function res = fn_find_max_old1(Af, M, H_s, vep, th0)

th = th0;  %given inital th

lr = 1e-4;

while 1
    U_th = H_s* diag( exp(-1i*th*vep) ) *H_s';   %encoding operator theta
    A_th = Af*U_th;
    
    %-------------near points------------
    %----[th_M2,th_M,th,th_P,th_P2]-----
    U_th_P = H_s* diag( exp(-1i*(th+lr)*vep) ) *H_s';
    A_th_P = Af*U_th_P;
    U_th_P2 = H_s* diag( exp(-1i*(th+lr+lr)*vep) ) *H_s';
    A_th_P2 = Af*U_th_P2;
    U_th_M = H_s* diag( exp(-1i*(th-lr)*vep) ) *H_s';
    A_th_M = Af*U_th_M;
    U_th_M2 = H_s* diag( exp(-1i*(th-lr-lr)*vep) ) *H_s';
    A_th_M2 = Af*U_th_M2;
    %-------------near points------------
    
    OverLap = fn_OverLap(A_th,A_th,M);
    OverLap_P = fn_OverLap(A_th_P,A_th_P,M);
    OverLap_P2 = fn_OverLap(A_th_P2,A_th_P2,M);
    OverLap_M = fn_OverLap(A_th_M,A_th_M,M);
    OverLap_M2 = fn_OverLap(A_th_M2,A_th_M2,M);
    
%     fprintf('th = %f, OverLap = %f, OverLap_M = %f, OverLap_P = %f \n',...
%              th,OverLap,OverLap_M,OverLap_P);
    
    tep1 = [OverLap_M2, OverLap_M, OverLap, OverLap_P, OverLap_P2];
    tep1 = real(tep1);
    [~,tep2] = max(tep1);  %= find(tep1 == max(tep1));
    
    if tep2 < 3
        th = th - lr;
    elseif tep2 > 3
        th = th + lr;
    elseif tep2 == 3
            break;
    end


    
%-------------check fn_OverLap_pd and fn_OverLap---------------
%     N = max(size(A0));
%     A_ini = eye(N);
%     tep3 = fn_OverLap_pd(A_ini, A_th, M);
%     tep4 = (OverLap_P - OverLap)/lr;
%     tep5 = (OverLap - OverLap_M)/lr;
% %     abs(tep3-tep4)
%     if abs(tep3-tep4)>1e-6 || abs(tep4-tep5)>1e-4
%        fprintf('Error of OverLap_pd = %f \n',abs(tep3-tep4))
%     end
%-------------check fn_OverLap_pd and fn_OverLap---------------
    

end

res = th;

end