% based on:
% https://ch.mathworks.com/matlabcentral/fileexchange/58320-demos-from--object-recognition--deep-learning--webinar/
%                             content/DeepLearningWebinar/Demo1_BagOfFeatures/Scene_Identification.m

clear; clc
addpath(genpath('./functions/'))
setDir = './data/2groups/original/';
vocSize = 350;
trainRatio = 10;
verbose = false;

% Load image data
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl=countEachLabel(imds);
categories = tbl.Label;
% minSetCount = min(tbl{:,2});



for i = 1:10

    [testSet,trainSet] = splitEachLabel(imds, trainRatio ,'randomized');
    
    %% Create Visual Vocabulary 
    bag = bagOfFeatures(trainSet, 'VocabularySize',vocSize, 'PointSelection','Detector', 'Verbose',false );


    word = double( encode(bag, trainSet, 'Verbose',false));
    trainData.X = array2table( word );
    trainData.Y = trainSet.Labels;
    trainData.categories = categories;

    %% Train classifier
    [ Classifier, ~ ] = trainMultiClassifier(trainData, 'Verbose', verbose );
    
    %% Test classifier 
    word = double( encode(bag, testSet, 'Verbose',false));
    testData.X = array2table( word );
    testData.Y = testSet.Labels;

    [label,~,~,score] = predict(Classifier,testData.X);

    correctPredictions = ( label == testData.Y );
    testAccuracy = sum(correctPredictions)/length(label);
    testAccuracy = floor(1000*testAccuracy)/10;

    acc(i) = testAccuracy;
    disp(mean(acc))

end

