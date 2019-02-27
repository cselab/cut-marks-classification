clc; clear


%% Options for CMA
opts.CMA.active = 0;
opts.PopSize = 100;
opts.Resume = 0;
opts.MaxFunEvals = 300000;
opts.LBounds = [-10  -10  -10]'; % upper bound of search space
opts.UBounds = [ 10   10   10]'; % lower bound of search space
opts.Noise.on = 0;
opts.LogModulo = 1;
opts.LogPlot = 1;
opts.EvalParallel = 0;
opts.EvalInitialX = 1;
opts.TolX = 1e-8;


% initial values
xinit = [5 5 5]';


fun = @(x) train_SVM_objective()
X = cmaes_parfor( 'fun',  xinit,[], opts);


X'
