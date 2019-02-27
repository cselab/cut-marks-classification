clear; clc
addpath(genpath('./functions/'))
setDir = './data/2groups/post_2/';
vocSize = 400;
trainRatio = 10;
verbose = 0;

% % Load image data

% localPath = "/Users/garampat/Documents/MATLAB/Machine Learning/bones classify/data/";
% remotePath = "./data";
% imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames', 'AlternateFileSystemRoots', [localPath, remotePath]);

imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl=countEachLabel(imds);
categories = tbl.Label;
% minSetCount = min(tbl{:,2});

% poolObj = gcp('nocreate');
% poolObj.addAttachedFiles('data/');

Ns = 4;
acc = zeros(Ns,1);

tt=tic();
parfor i = 1:Ns

    fprintf('%d\n',i);
    
    [testSet,trainSet] = splitEachLabel(imds, trainRatio ,'randomized');
    
    
    %% Create Visual Vocabulary 
    bag = bagOfFeatures(trainSet, 'VocabularySize',vocSize, 'PointSelection','Detector', 'Verbose',verbose );
%     bag = bagOfFeatures(trainSet, 'VocabularySize',vocSize, 'PointSelection','Grid','GridStep',[4,4], 'Verbose',false );

    fprintf('%d\n',i);

    word = double( encode(bag, trainSet, 'Verbose',false));
    trainData = struct();
    trainData.X = my_array2table( word );
    trainData.Y = trainSet.Labels;
    trainData.categories = categories;

    % Train classifier
    [ Classifier, ~ ] = trainClassifier(trainData, 'Verbose', verbose );
%     [ Classifier, ~ ] = trainClassifier_CMA(trainData, 'Verbose', false);

    %% Test classifier 
    word = double( encode(bag, testSet, 'Verbose',false));
    testData = struct();
    testData.X = my_array2table( word );
    testData.Y = testSet.Labels;

    [label,score] = Classifier.predictPostFcn(testData.X);

    correctPredictions = ( label == testData.Y );
    testAccuracy = sum(correctPredictions)/length(label);
    testAccuracy = floor(1000*testAccuracy)/10;

    acc(i) = testAccuracy;
    
end

%%
fprintf('Elaplsed time: %f\n',toc(tt))
fprintf('\n Mean accuracy: %.2f%%   with std: %.2f%% \n\n',mean(acc) , std(acc))



