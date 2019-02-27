clear

% data1=load('data_surf_400_a.mat');
% data2=load('data_surf_400_b.mat');
% accuracy = [ data1.accuracy ; data2.accuracy];

data1=load('data_SURF_600.mat');
accuracy = [ data1.accuracy ];


acc = mean(accuracy,2);


figure(1); clf
plot(accuracy,'-o')
legend(string(data1.categories(1)),string(data1.categories(2)))
grid on

figure(2); clf
plot(cumsum(accuracy)./repmat((1:length(accuracy))',1,2),'-o'); hold on
plot(cumsum(acc)./(1:length(acc))','-o')
legend(string(data1.categories(1)),string(data1.categories(2)),'total')
grid on



[ mean(100*accuracy(:,1)) std(100*accuracy(:,1)); mean(100*accuracy(:,2)) std(100*accuracy(:,2)); mean(100*acc) std(100*acc)]

figure(3); clf
subplot(1,3,1)
histogram(accuracy(:,1),20)
title(string(data1.categories(1)))

subplot(1,3,2)
histogram(accuracy(:,2),20)
title(string(data1.categories(2)))

subplot(1,3,3)
histogram(acc,20)
title('total')