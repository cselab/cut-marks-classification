function y = train_SVM_objective( par, data )


method.name   = data.KernelFunction;
method.PolOrd = data.PolynomialOrder;
method.Scale  = 10^par(1);
method.BoxCon = 10^par(2);


[ ~, y ] = train_SVM( data.trainData, method);

y = 1-y;