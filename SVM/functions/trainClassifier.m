function [ Classifier, Accuracy] = trainClassifier( varargin )

p = inputParser;

p.addRequired( 'trainData' );
p.addParameter('Verbose', false );

p.parse( varargin{:} );



trainData = p.Results.trainData;
verbose   = p.Results.Verbose;
clear p;

% methodInfo = {  {'linear',[], 'auto', 1 }, ...
%                 {'poly'  , 2, 'auto', 1 }, ...
%                 {'poly'  , 3, 'auto', 1 }, ...
%                 {'gauss' ,[],   4   , 1 }, ...
%                 {'gauss' ,[],   6   , 1 },...
%                 {'gauss' ,[],   8   , 1 },...
%                 {'gauss' ,[],  10   , 1 },...
%                 {'gauss' ,[],  12   , 1 },...
%                 {'gauss' ,[],  14   , 1 },...
%                 {'gauss' ,[],  16   , 1 },...
%                 {'gauss' ,[],  18   , 1 },...
%                 {'gauss' ,[],  32   , 1 },...
%                 {'gauss' ,[],  64   , 1 },...
%                 {'gauss' ,[],  96   , 1 } };

methodInfo = {  {'gauss-auto',[],[],[] } };

                

nNames = numel(methodInfo);
trainedClassifier  = cell(nNames,1);
validationAccuracy = zeros(nNames,1);


for i=1:nNames
    
    method.name   = methodInfo{i}{1};
    method.PolOrd = methodInfo{i}{2};
    method.Scale  = methodInfo{i}{3};
    method.BoxCon  = methodInfo{i}{4};
    method.Verbose = verbose;
    method.ShowPlots = false;
    
    [trainedClassifier{i}, validationAccuracy(i)] = train_SVM( trainData, method );
    validationAccuracy(i) = floor(1000*validationAccuracy(i))/10;
    
    if(verbose)
        printInfo
    end

end

% [~,ind] = max(validationAccuracy);
ind = find( validationAccuracy==max(validationAccuracy),1,'last' ) ;

% if verbose
%     printWinner
% end


Classifier = trainedClassifier{ind};
Accuracy   = validationAccuracy(ind);


    



    function printInfo
        switch method.name
            
            case {'linear','quad','poly'}
                fprintf('\n   %2d) SVM trained with kernel: %8s          and accuracy:   %3.1f%% ', i, method.name, validationAccuracy(i))
                
            case {'gauss'}
                fprintf('\n   %2d) SVM trained with kernel: %8s (%3d)    and accuracy:   %3.1f%% ', i, method.name, method.param, validationAccuracy(i))
                
        end
    end



    function printWinner
        switch methodInfo{ind}{1}
            
            case {'linear','quad'}
                fprintf('\n\n   *** SVM with kernel %8s         wins! ***\n ', methodInfo{ind}{1} )
                
            case {'poly','gauss'}
                fprintf('\n\n   *** SVM with kernel %8s (%3d)    wins! ***\n ', methodInfo{ind}{1}, methodInfo{ind}{2} )
                
        end
    end




end
