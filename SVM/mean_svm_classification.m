clear; clc
addpath(genpath('./functions/'))
setDir = './data/2groups/post_2/';
vocSize = 400;
trainRatio = 10;
verbose = 1;

% Load image data
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl=countEachLabel(imds);
categories = tbl.Label;

Ns = 60;
acc = zeros(Ns,1);

tt=tic();
for i = 1:Ns

    fprintf('%d\n',i);
    
    [testSet,trainSet] = splitEachLabel(imds, trainRatio ,'randomized');
    
    
    %% Create Visual Vocabulary 
%     bag = bagOfFeatures(trainSet, 'VocabularySize',vocSize, 'PointSelection','Detector', 'Verbose', false );
    bag = bagOfFeatures(trainSet, 'VocabularySize',vocSize, 'PointSelection','Grid','GridStep',[8,8], 'Verbose',false );

    fprintf('%d\n',i);

    word = double( encode(bag, trainSet, 'Verbose',false) );
    trainData = struct();
    trainData.X = array2table( word );
    trainData.Y = trainSet.Labels;
    trainData.categories = categories;

    % Train classifier
    [ Classifier, ~ ] = trainClassifier(trainData, 'Verbose', verbose );
%     [ Classifier, ~ ] = trainClassifier_CMA(trainData, 'Verbose', false);

    %% Test classifier 
    word = double( encode(bag, testSet, 'Verbose',false));
    testData = struct();
    testData.X = array2table( word );
    testData.Y = testSet.Labels;

    [label,score] = Classifier.predictPostFcn(testData.X);

    fprintf('Accuracy: \n');
    for j=1:length(categories)
        ind =  ( testData.Y==categories(j) );
        sz = sum(ind);

        accuracy(i,j) = sum( label(ind) == testData.Y(ind) )/sz;
    end
    
    acc(i) = mean(accuracy(i,:));
    
    fprintf('\n Mean accuracy: %.2f   with std: %.2f \n\n',mean(100*acc(1:i)) , std(100*acc(1:i)))
    
    save('data_post2.mat')
    
end

%%
fprintf('Elaplsed time: %f\n',toc(tt))