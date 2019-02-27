
clear;
addpath(genpath('./functions/'))
setDir = './data/2groups/post_1_1/';
vocSize = 400;
trainRatio = 1;
verbose = 1;

% Load image data


imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');

%%

[testSet,trainSet] = splitEachLabel(imds, trainRatio ,'randomized');



%%
tbl=countEachLabel(imds);
categories = tbl.Label;
minSetCount = min(tbl{:,2});

% rng(123)


if verbose
    cprintf('*blue','Label \t\t Count \t\t Train Count \t\t Test Count '); fprintf('\n');
    for i=1:numel(categories)
        
        cprintf('blue','%s \t\t   %d \t\t      %d \t\t     %d',...
                                categories(i), sum(imds.Labels==categories(i)),...
                                sum(trainSet.Labels==categories(i)),...
                                sum(testSet.Labels==categories(i)) ); 
        fprintf('\n');
    end
end



%% Create Visual Vocabulary 
fprintf('\nCreating Vocabulary of size %d \n\n',vocSize);
bag = bof(trainSet, 'VocabularySize',vocSize, 'PointSelection','Detector', 'Verbose',false );
% bag = bagOfFeatures(trainSet, 'VocabularySize',vocSize, 'PointSelection','Grid','GridStep',[4,4], 'Verbose',false );


% vis_feature_vectors(trainingSet, bag)

word = double( encode(bag, trainSet, 'Verbose',false));
trainData.X = array2table( word );
trainData.Y = trainSet.Labels;
trainData.categories = categories;



%% Train classifier
% classificationLearner; return;
fprintf('Training the classifier... ')
[ Classifier, ~ ] = trainClassifier(trainData, 'Verbose', verbose );
% [ Classifier, ~ ] = trainClassifier_CMA(trainData, 'Verbose', false);
fprintf('\n\n');



%% Test classifier 

word = double( encode(bag, testSet, 'Verbose',false));
testData.X = array2table( word );
testData.Y = testSet.Labels;

fprintf('Test the classifier... ')

[label,score] = Classifier.predictPostFcn(testData.X);


fprintf('Accuracy: \n');
for i=1:length(categories)
    ind =  ( testData.Y==categories(i) );
    sz = sum(ind);
    
    accuracy(i) = sum( label(ind) == testData.Y(ind) )/sz;
    
    tmp = floor(1000*accuracy(i))/10;
    fprintf('  in label %s : %3.1f%%\n',categories(i),tmp)
    
end


correctPredictions = ( label == testData.Y );
testAccuracy = sum(correctPredictions)/length(label);
testAccuracy = floor(1000*testAccuracy)/10;

fprintf('overall accuracy %3.1f%%\n\n', testAccuracy)

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



save('tmp.mat');