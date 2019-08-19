layerSizes = [400 25 10];

% Initialize an unrolled neural network
fprintf('\nInitializing Neural Network... \n');
neuralNetwork = initNeuralNetwork(layerSizes);

% Load training data
fprintf('\nLoading training data... \n');
load('trainingSet.mat');
load('debugWeights.mat');
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