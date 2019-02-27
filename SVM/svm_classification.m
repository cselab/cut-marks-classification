

clear;
addpath(genpath('./functions/'))
setDir = './data/2groups/post_1/';
vocSize = 600;
trainRatio = 0;
verbose = 1;
pcrtFtr = 0.8;

% Load image data
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl=countEachLabel(imds);
categories = tbl.Label;

% rng(123)
trainSet = imds;


%% Create Visual Vocabulary 
fprintf('\nCreating Vocabulary of size %d \n\n',vocSize);
bag = bagOfFeatures(trainSet,...
                    'VocabularySize', vocSize,...
                    'PointSelection','Detector',...
                    'Verbose',true );
                
% bag = bagOfFeatures( trainSet,...
%                     'VocabularySize',vocSize,...
%                     'PointSelection','Grid',...
%                     'GridStep',[12,12],...
%                     'Verbose',true,...
%                     'StrongestFeatures', pcrtFtr );

% vis_feature_vectors(trainSet, bag)


word = double( encode(bag, trainSet, 'Verbose',true));
trainData.X = array2table( word );
trainData.Y = trainSet.Labels;
trainData.categories = categories;



%% Train classifier
% classificationLearner; return;
fprintf('Training the classifier... ')
[ Classifier, ~ ] = trainClassifier(trainData, 'Verbose', verbose);
% [ Classifier, ~ ] = trainClassifier_CMA(trainData, 'Verbose', false);
fprintf('\n\n');



%% Predict
setDir = './data/uncategorized/arc_post_1/';
imds_pred = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

word = double( encode(bag, imds_pred, 'Verbose',true));

X = array2table( word );
[label,score] = Classifier.predictPostFcn(X);

