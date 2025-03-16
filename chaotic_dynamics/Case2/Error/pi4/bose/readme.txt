质量比为1的情形。

文件 equal_mass_diedai.m
（两体等质量）已知c=无穷时的玻色子(准动量)解，通过BA方程的迭代，
得到任意相互作用的(准动量)解；费米情形下解不随相互作用改变。
费米解+玻色解，得到完全的解。


文件 equal_mass_limit.m
（两体等质量）求解c=0和c=无穷时的解


文件 equal_mass_fsolve.m
（两体等质量）已知c=无穷时的玻色子(准动量)解，通过fsolve函数迭代，
得到任意相互作用的(准动量)解。（可以正常迭代不推荐用此方法）


数据 equal_BA_zero.mat
等质量，c=0 时的解，由文件equal_mass_limit.m所得到。


数据 equal_BA_inf.mat
等质量，c=inf 时的解，此时能级是二重简并，由文件equal_mass_limit.m所得到。


数据 equal_BA_c2.mat - equal_BA_c5.mat
分别c=1，10，100，10^12 时的bose解，由文件 equal_mass_diedai.m 得到。
