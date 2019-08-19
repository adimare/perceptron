% Init 
layerSizes = [400 25 10];
lambda = 1;
maxIterations = 50;

% Initialize neural network with random weights
fprintf('\nInitializing Neural Network... \n');
neuralNetwork = initNeuralNetwork(layerSizes);

% Load training data from file (sets X and y)
fprintf('\nLoading training data... \n');
load('trainingSet.mat');

% Train neural network
fprintf('\nTraining neural network... \n');
neuralNetwork = train(neuralNetwork, layerSizes, X, y, lambda, maxIterations);

% Test trained neural network
predictions = zeros(size(X,1), 1);
for i=1:size(X,1)
  predictions(i) = predict(neuralNetwork, layerSizes, X(i,:));
end
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predictions == y)) * 100);
