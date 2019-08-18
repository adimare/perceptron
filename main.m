numLayers = 5;
layerSizes = [3 7 5 6 12];

% Initialize an unrolled neural network
neuralNetwork = initNeuralNetwork(layerSizes);

input = [1 2 3];
predict(input, neuralNetwork, layerSizes);