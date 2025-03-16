这里选取的满足偶宇称的高斯波包作为初态，在完整哈密顿量H(odd+even)下演化，进行计算回波。


文件 echo_no2_c1.m - echo_no2_c4.m 
分别计算质量比1/3，相互作用c=0,1,10,100时的回波演化行为，
需要用到子文件 fn_FEM_matrix.m或其产生的数据 matrix_echo_no2.mat。


子文件 fn_FEM_matrix.m
次子文件 Gradients.m，  InterMat2D.m，  StiffMat2D.m，  MassMat2D.m
计算薛定谔方程的有限元矩阵A，M，K，以及偶宇称下所对应的矩阵A_even，M_even，K_even，
奇宇称下所对应的矩阵A_odd，M_eodd，K_odd。大概3w*3w左右的稀疏矩阵


文件 plot_Echo_no2.m 
用数据data_echo_no2_c1.mat - data_echo_no2_c4.mat 画各个相互作用下的回波演化的图。


数据 matrix_echo_no2.mat
子文件 fn_FEM_matrix.m所产生的数据。


数据 data_echo_no2_c1.mat - data_echo_no2_c4.mat 
分别为文件 echo_no2_c1.m - echo_no2_c4.m 产生的数据。
