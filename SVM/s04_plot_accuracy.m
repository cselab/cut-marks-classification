% Plots the accuracy results obtained from running 02_mean_svm_test_classification.m
%
% author: George Arampatzis (garampat@ethz.ch)

clear


data1=load('data_SURF_600.mat');
accuracy = [ data1.accuracy ];

% data1 = load('data_post2.mat');
% accuracy = [ data1.accuracy ];

acc = mean(accuracy,2);


%%
figure(1); clf
p = plot(accuracy,'-o','LineWidth',3,'MarkerSize',10);
l=legend( string(data1.categories(1)), string(data1.categories(2)) );
l.Location = 'best';
grid on
set(gca,'FontSize',18)
xlabel('# run')
ylabel('accuracy')


%%
figure(2); clf
plot( cumsum(accuracy)./repmat((1:length(accuracy))',1,2),'-o','LineWidth',3,'MarkerSize',10); 
hold on
plot( cumsum(acc)./(1:length(acc))','-o','LineWidth',3,'MarkerSize',10 )
l = legend(string(data1.categories(1)),string(data1.categories(2)),'total');
l.Location = 'best';
grid on
set(gca,'FontSize',18)
xlabel('# run')
ylabel('running mean accuracy')
axis tight
