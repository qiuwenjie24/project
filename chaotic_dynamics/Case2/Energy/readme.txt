

子文件 fn_FEM_matrix.m
计算薛定谔方程的有限元矩阵A，M，K，以及偶宇称下所对应的矩阵A_even，M_even，K_even，
奇宇称下所对应的矩阵A_odd，M_eodd，K_odd。


子文件 Gradients.m，  InterMat2D.m，  StiffMat2D.m，  MassMat2D.m
分别有限元中需要的一个梯度函数和组装有限元矩阵A，M，K的函数。


文件 Eigenvalue_no1.m - Eigenvalue_no4.m 
分别计算质量比1,1/3等四种质量比的有限元矩阵，以及c=0,1,10,100,10^12的前2000个偶宇称的能级。
数据在 文件夹no1-no4 中。


文件 Eigenvalue_no2_2.m - Eigenvalue_no4_2.m 
分别计算质量比1,1/3等四种质量比的有限元矩阵，以及c=0,1,10,100,10^12的前2000个奇宇称的能级
和全部(奇偶宇称)能级。
数据在 文件夹no1_2 - no4_2 中。


数据 Matrix_M_A_N_no1.mat - Matrix_M_A_N_no4.mat
分别为文件 Eigenvalue_no1 - Eigenvalue_no4.m 中通过子文件 fn_FEM_matrix.m产生的有限元矩阵。





