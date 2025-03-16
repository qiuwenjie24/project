%% Microcanonical OTOC
clear

%% theta=pi/4
figure
load('mic_OTOC_pi4.mat') 

time=0:0.001:3;
plot(time,c_n_350,'Color',[0.4940 0.1840 0.5560],'LineWidth',2)
hold on
plot(time,c_n_400,'Color',[0.9290 0.6940 0.1250],'LineWidth',2)
hold on
plot(time,c_n_450,'Color',[0.8500 0.3250 0.0980],'LineWidth',2)
hold on
plot(time,c_n_500,'Color',[0 0.4470 0.7410],'LineWidth',2)

xlabel('t')
ylabel('$c_{200}(t)$','interpreter','latex');
legend('$N_{trunc}=350$','$N_{trunc}=400$','$N_{trunc}=450$','$N_{trunc}=500$',...
    'interpreter','latex','fontsize',17);

set(gca,'FontSize',20,'linewidth',2.5) %坐标轴 线粗0.5磅
set(get(gca,'XLabel'),'FontSize',28); %x轴文字
set(get(gca,'YLabel'),'FontSize',28);
axis([0 3 0 500])

