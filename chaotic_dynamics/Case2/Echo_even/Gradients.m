%%StiffMat2D里面的梯度函数
function [area,b,c] = Gradients(x,y)
area=polyarea(x,y);   %3个节点围成区间的面积
b=[y(2)-y(3); y(3)-y(1); y(1)-y(2)]/2/area;   %gradient of phi_i= [b_i ;c_i ] 
c=[x(3)-x(2); x(1)-x(3); x(2)-x(1)]/2/area;