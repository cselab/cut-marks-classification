% based on:
% https://ch.mathworks.com/matlabcentral/fileexchange/58320-demos-from--object-recognition--deep-learning--webinar/
%    content/DeepLearningWebinar/Demo1_BagOfFeatures/Scene_Identification.m

clear; 
addpath(genpath('./functions/'))
setDir = './data/2groups/post_1/';
vocSize = 200;
trainRatio = 10;
verbose = true;

% Load image data
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
tbl=countEachLabel(imds);
categories = tbl.Label;
minSetCount = min(tbl{:,2});

rng(123)
[testSet,trainSet] = splitEachLabel(imds, trainRatio ,'randomized');

if verbose
    cprintf('*blue','Label \t\t Count \t\t Train Count \t\t Test Count '); fprintf('\n');
    for i=1:numel(categories)
        
        cprintf('blue','%s \t\t   %d \t\t      %d \t\t     %d',categories(i), sum(imds.Labels==categories(i)), sum(trainSet.Labels==categories(i)), sum(testSet.Labels==categories(i)) ); 
        fprintf('\n');
    end
end



%% Create Visual Vocabulary 
fprintf('\nCreating Vocabulary of size %d \n\n',vocSize);
bag = bagOfFeatures(trainSet, 'VocabularySize',vocSize, 'PointSelection','Detector', 'Verbose',false );

%% Train classifier
fprintf('Training the classifier... ')
opts = templateSVM('KernelFunction', 'polynomial', ...
                    'PolynomialOrder', 3, ...
                    'KernelScale', 'auto', ...
                    'BoxConstraint', 1, ...
                    'Standardize', true);
classifier = trainImageCategoryClassifier(trainSet,bag, 'LearnerOptions', opts);
fprintf('\n\n');




evaluate(classifier,testSet);

