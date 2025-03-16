质量比1/3情形。
注意：由文件 unequal_mass_finite_odd.m，unequal_mass_finite_even.m 
          所得到的能级(BAroots_finite_odd_c100.mat,BAroots_finite_even_c100.mat )，
          是没有进行能级大小排序的。

子文件 fun_BAE_o.m，fun_BAE_e.m
分别为奇宇称BA方程 和 偶宇称BA方程 的函数。


文件 unequal_mass_finite_odd.m
需要用到文件 fun_BAE_o.m，和数据 unequal_BA_inf.mat，
通过知道c=无穷时的解，用fsolve函数迭代求解到c=100的奇宇称解。


文件 unequal_mass_finite_even.m
需要用到文件 fun_BAE_e.m，和数据 unequal_BA_inf.mat，
通过知道c=无穷时的解，用fsolve函数迭代求解到c=100的偶宇称解。


文件 unequal_mass_inf.m
计算质量为1/3相互作用为无穷时候的BA解，此时奇宇称与偶宇称的解一致，二重简并，这里只求了一种。


文件 unequal_mass_zero.m
计算质量为1/3相互作用c=0 时候的BA解。


数据 unequal_BA_inf.mat
质量为1/3相互作用为无穷时候的BA解，由文件 unequal_mass_inf.m所得。


数据 BAroots_finite_odd_c100.mat  
计算质量为1/3相互作用为c=100时候的奇宇称的BA解，由文件 unequal_mass_finite_odd.m所得。


数据 BAroots_finite_even_c100.mat  
计算质量为1/3相互作用为c=100时候的偶宇称的BA解，由文件 unequal_mass_finite_even.m所得。


