这里计算的是整个哈密顿量的OTOC，无法计算偶宇称的的OTOC，原因是OTOC需要用到了算符X,P，
只用部分空间会导致[X,P]对易关系改变，从而出问题。


文件 OTOC_no2_c1.m - OTOC_no2_c5.m 
用有限元方法计算质量比1/3，相互作用分别为c=0,1,10,100,10^12的OTOC的演化。
需要用到 子文件 fn_Q.m,  fn_FEM_x_Q.m,  fn_Cc.m,  fn_c.m,  fn_b.m
数据在 文件夹no2 中，c1-c5文件分别对应不同的相互作用的OTOC的数据。


子文件 Gradients.m，  InterMat2D.m，  StiffMat2D.m，  MassMat2D.m
分别有限元中需要的一个梯度函数和组装有限元矩阵A，M，K的函数。


文件 plot_ther_OTOC.m
画质量比1/3，相互作用分别为c=0,1,10,100 的thermal OTOC的演化，
用到了文件夹no2\c1 - c5 中的数据 ther_OTOC_all_no2_c1.mat - ther_OTOC_all_no2_c4.mat


文件 plot_mic_OTOC.m
画质量比1/3，相互作用分别为c=0,1,10,100 的micro OTOC的演化，
用到了文件夹no2\c1 - c5 中的数据 mic_OTOC_all_no2_c1.mat - mic_OTOC_all_no2_c4.mat


文件 plot_OTOC_c5.m
画质量比1/3，相互作用分别为c=10^12 的micro OTOC和thermal OTOC的演化，
用到了文件夹no2\c1 - c5 中的数据 ther_OTOC_all_no2_c5.mat 和 mic_OTOC_all_no2_c5.mat


文件no2的数据：
c1-c5文件夹分别储存了相互作用分别为c=0,1,10,100,10^12 的数据。
以c1文件夹为例，
State_all_no2_c1.mat 储存 本征态和本征值
matrix_OTOC_all_no2_c1.mat 储存计算OTOC需要到的矩阵，可以用来检测X,P的对易关系，
mic_OTOC_all_no2_c1.mat 储存 micro OTOC数据
ther_OTOC_all_no2_c1.mat 储存 thermal OTOC数据

