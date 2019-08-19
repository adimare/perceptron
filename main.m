layerSizes = [400 25 10];

% Initialize an unrolled neural network
fprintf('\nInitializing Neural Network... \n');
neuralNetwork = initNeuralNetwork(layerSizes);

% Load training data
fprintf('\nLoading training data... \n');
load('ex4data1.mat');

options = optimset('MaxIter', 50);
lambda = 1;
% Create "short hand" for the cost function to be minimized
minCostFunction = @(p) costFunction(p, layerSizes, X, y, lambda);

[neuralNetwork, cost] = fmincg(minCostFunction, neuralNetwork, options);

predictions = zeros(size(X,1), 1);
for i=1:size(X,1)
  predictions(i) = predict(neuralNetwork, X(i,:), layerSizes);
end

fprintf('\nTraining Set Accuracy: %f\n', mean(double(predictions == y)) * 100);

if test==1
  load('ex4weights.mat');
  % Unroll parameters 
  neuralNetwork = [Theta1(:) ; Theta2(:)];
  lambda = 0;
  J = costFunction(neuralNetwork, layerSizes, X, y, lambda);

  fprintf(['Cost at parameters (loaded from ex4weights): %f '...
          '\n(this value should be about 0.287629)\n'], J);

  lambda = 1;
  J = costFunction(neuralNetwork, layerSizes, X, y, lambda);

  fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

  checkNNGradients;
else
  
end