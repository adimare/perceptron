layerSizes = [400 25 10];

% Initialize an unrolled neural network
% neuralNetwork = initNeuralNetwork(layerSizes);

% input = [1 2 3];
% predict(input, neuralNetwork, layerSizes);

% Load crap
load('ex4data1.mat');
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