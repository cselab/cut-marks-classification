function [Classifier, validationAccuracy] = trainClassifier_CMA( varargin )

                        

p = inputParser;

p.addRequired( 'trainData' );
p.addParameter('KernelFunction', 'gauss' );
p.addParameter('PolynomialOrder', [] );
p.addParameter('Verbose', false );

p.parse( varargin{:} );




data.trainData       = p.Results.trainData;
data.KernelFunction  = p.Results.KernelFunction;
data.PolynomialOrder = p.Results.PolynomialOrder;

verbose   = p.Results.Verbose;

clear p;






%% Options for CMA
if( verbose )
    opts.DispModulo = 1;
    opts.LogPlot = 1;
    opts.DispFinal = 'on';
else
    opts.DispModulo = 0;
    opts.LogPlot = 0;
    opts.DispFinal = 'off';
end

opts.CMA.active = 0;
opts.PopSize = 100;
opts.Resume = 0;
opts.MaxFunEvals = 5000;
opts.LBounds = [-3  -3 ]'; % lower bound of search space
opts.UBounds = [ 3   3 ]'; % upper bound of search space
opts.Noise.on = 0;
opts.EvalParallel = 1;
opts.EvalInitialX = 1;
opts.TolX = 1e-8;
opts.LogModulo = 1;


% initial values
xinit = [0 0]';

par = cmaes_parfor( 'train_SVM_objective',  xinit, [], opts, data);


if(verbose)
    disp( 10.^par' );
end
%% obtain the classifier at the optimized hyperparameters

method.name   = data.KernelFunction;
method.PolOrd = data.PolynomialOrder;
method.Scale  = 10^par(1);
method.BoxCon = 10^par(2);

[Classifier, validationAccuracy] = train_SVM( trainData, method);

