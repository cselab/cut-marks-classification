function vis_feature_vectors(trainingSet, bag)

% Visualize Feature Vectors 
human = find(trainingSet.Labels == 'human');
nature = find(trainingSet.Labels == 'nature');
%
figure(1)
img = imread(   trainingSet.Files{ human(randperm(numel(human),1) ) } ) ;
featureVector = encode(bag, img);
subplot(2,2,1); imshow(img);
subplot(2,2,2); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = imread(   trainingSet.Files{ nature(randperm(numel(nature),1) ) } ) ;
featureVector = encode(bag, img);
subplot(2,2,3); imshow(img);
subplot(2,2,4); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');