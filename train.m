% Trains a neural network of size determined by layerSizes using the training data given by X and results given by y
% Lambda is the regularization parameter, maxIterations sets a limit for the amount of times backpropagation is executed
function neuralNetwork = train(neuralNetwork, layerSizes, X, y, lambda, maxIterations)

  % Create "short hand" for the cost function to be minimized
  minCostFunction = @(p) costFunction(p, layerSizes, X, y, lambda);
  options = optimset('MaxIter', 50);
  [neuralNetwork, cost] = fmincg(minCostFunction, neuralNetwork, options);

end
