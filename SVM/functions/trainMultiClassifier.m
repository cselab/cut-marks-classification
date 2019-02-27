function [ Classifier, Accuracy] = trainMultiClassifier( varargin )

p = inputParser;

p.addRequired( 'trainData' );
p.addParameter('Verbose', true );

p.parse( varargin{:} );


trainData = p.Results.trainData;
verbose   = p.Results.Verbose;
clear p;




t = templateSVM('Standardize',1,'KernelFunction','gaussian');

Classifier = fitcecoc(  trainData.X,trainData.Y,...
                        'Learners',t,...
                        'ClassNames',trainData.categories,...
                        'FitPosterior',1,...
                        'OptimizeHyperparameters','auto',...
                        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));

% 'Verbose',1,...

                    
Accuracy = resubLoss(Classifier);

end
