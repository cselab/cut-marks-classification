% based on:
% https://ch.mathworks.com/matlabcentral/fileexchange/58320-demos-from--object-recognition--deep-learning--webinar/
%    content/DeepLearningWebinar/Demo1_BagOfFeatures/Scene_Identification.m

clear; clc
addpath(genpath('./functions/'))
setDir = './data/2groups/original/';
vocSize = 350;
trainRatio = 0.75;
verbose = true;

% Load image data
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl=countEachLabel(imds);
categories = tbl.Label;
minSetCount = min(tbl{:,2});

% [trainSet,testSet] = splitEachLabel(imds, minSetCount-10 ,'randomized');
[testSet,trainSet] = splitEachLabel(imds, 10 ,'randomized');

if verbose
    cprintf('*blue','Label    \t\t Count \t\t Train Count \t\t Test Count '); fprintf('\n');
    for i=1:numel(categories)
        
        cprintf('blue','%s    \t\t   %d \t\t      %d \t\t     %d',categories(i), sum(imds.Labels==categories(i)), sum(trainSet.Labels==categories(i)), sum(testSet.Labels==categories(i)) ); 
        fprintf('\n');
    end
end



%% Create Visual Vocabulary 
fprintf('\nCreating Vocabulary of size %d \n\n',vocSize);
bag = bagOfFeatures(trainSet, 'VocabularySize',250, 'PointSelection','Detector', 'Verbose',false );

% vis_feature_vectors(trainingSet, bag)

word = double( encode(bag, trainSet, 'Verbose',false));
trainData.X = array2table( word );
trainData.Y = trainSet.Labels;
trainData.categories = categories;

% trainData = double( encode(bag, trainSet, 'Verbose',false));
% trainData = array2table(trainData);
% trainData.dataType = trainingSet.Labels;


%% Train classifier
% classificationLearner; return;
fprintf('Training the classifier... ')
[ Classifier, ~ ] = trainMultClassifier(trainData, 'Verbose', verbose );
fprintf('\n\n');



%% Test classifier 
word = double( encode(bag, testSet, 'Verbose',false));
testData.X = array2table( word );
testData.Y = testSet.Labels;

fprintf('Test the classifier... ')

[label,~,~,score] = predict(Classifier,testData.X);

correctPredictions = ( label == testData.Y );
testAccuracy = sum(correctPredictions)/length(label);
testAccuracy = floor(1000*testAccuracy)/10;

fprintf('with accuracy %3.1f%%\n\n', testAccuracy)

%%

if(verbose)
    prb = score( sub2ind(size(score), 1:size(score,1), grp2idx(label)') );
    prb = floor(1000*prb)/10;

    fprintf('\n');
    cprintf('*black',' True Labels \t\t Predicted Labels \t Probability')
    fprintf('\n');
    for i=1:numel(prb)

        if(testData.Y(i)==label(i))
            col = 'black';
        else
            col = 'red';
        end

        cprintf(col, '%8s \t\t %8s \t\t    %3.1f%%', testData.Y(i), label(i), prb(i)  );
        fprintf('\n');

    end
    fprintf('\n')
end





%% Visualize how the classifier works
% jj = find(correctPredictions==0,1,'first');
% 
% if( ~isempty(jj) )
%     
%     img = imread( testSet.Files{jj} );
% 
%     figure(2); clf
%     imshow(img)
%     imagefeatures = double(encode(bag, img));
%     % Find two closest matches for each feature
%     [bestGuess, score] = predict( trainedClassifier.ClassificationSVM, imagefeatures );
%     % Display the string label for img
%     if bestGuess==testSet.Labels(jj)
%         titleColor = [0 0.8 0];
%     else
%         titleColor = 'r';
%     end
%     title(sprintf('Best Guess: %s; Actual: %s',...
%         char(bestGuess),char(testSet.Labels(jj)) ),...
%         'color',titleColor)
% end
