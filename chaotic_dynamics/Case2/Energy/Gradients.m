%%StiffMat2D������ݶȺ���
function [area,b,c] = Gradients(x,y)
area=polyarea(x,y);   %3���ڵ�Χ����������
b=[y(2)-y(3); y(3)-y(1); y(1)-y(2)]/2/area;   %gradient of phi_i= [b_i ;c_i ] 
c=[x(3)-x(2); x(1)-x(3); x(2)-x(1)]/2/area;