质量比为1的情形。

文件 equal_mass_finite_odd.m
（两体等质量）已知c=无穷时的玻色子(准动量)解，通过fsolve函数迭代，
得到c=100的奇宇称的(准动量)解。


文件 equal_mass_finite_even.m
（两体等质量）已知c=无穷时的玻色子(准动量)解，通过fsolve函数迭代，
得到c=100的偶宇称的(准动量)解。


上面两个文件需要用到文件  '.\bose\equal_mass_limit.m'  中得到的（c=无穷时的解）数据 equal_BA_inf.mat


fun_BAE_e.m
偶宇称的BA方程，文件 equal_mass_finite_even.m 所需要。


fun_BAE_o.m
奇宇称的BA方程，文件 equal_mass_finite_odd.m 所需要。