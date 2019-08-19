# Perceptron
A configurable multiclass neural network compatible with MATLAB and Octave.

# initNeuralNetwork
Used to initialize a neural network with random weights.

|Param|Type|Values|
|---|---|---|
|**layerSizes**|*Required* array|Determines the amount of layers in the neural network and the size of each layer. The first number in the array corresponds to the size of the input layer, the last number corresponds to the size of the output layer, and all numbers in between determine the sizes of the hidden layers|

```console
% Initialize a neural network with 400 neurons in the input layer, two hidden layers of 16 neurons, and an output layer of 10 neurons.
neuralNetwork = initNeuralNetwork([400 16 16 10]);
```

# train
Implements the fmincg algorithm to trains a neural network using the training data given by X and y.

|Param|Type|Values|
|---|---|---|
|**neuralNetwork**|*Required* array|An unrolled neural network's theta values.|
|**layerSizes**|*Required* array|Determines the amount of layers in the neural network and the size of each layer. The first number in the array corresponds to the size of the input layer, the last number corresponds to the size of the output layer, and all numbers in between determine the sizes of the hidden layers|
|**X**|*Required* matrix|The training data. Each row corresponds to a training example. The number of columns must match the size of the input layer represented in the first number of the layerSizes array.|
|**y**|*Required* array|The labels that correspond to the training data.|
|**lambda**|*Optional* integer|Used for regularization to avoid overfitting. If the neural network performs too well on the training set but not so much on validation/test sets, the lambda value should be increased. Defaults to 0.|
|**maxIterations**|*Optional* integer|Determines the max amount of times the backpropagation algorithm is executed to update the neural network's weights. Defaults to 50.|

```console
neuralNetwork = train(neuralNetwork, [400 16 16 10], X, y, 1, 100);
```
# predict
Executes forward propagation on a neural network to produce a prediction based on a test set.
|Param|Type|Values|
|---|---|---|
|**neuralNetwork**|*Required* array|An unrolled neural network's theta values.|
|**layerSizes**|*Required* array|Determines the amount of layers in the neural network and the size of each layer. The first number in the array corresponds to the size of the input layer, the last number corresponds to the size of the output layer, and all numbers in between determine the sizes of the hidden layers|
|**input**|*Required* array|The features for an unlabeled example.|

```console
result = predict(neuralNetwork, [400 16 16 10], TestSet(12,:));
```
#usage example
```console
% Init 
layerSizes = [400 25 10];
lambda = 1;
maxIterations = 50;

% Initialize neural network with random weights
fprintf('\nInitializing Neural Network... \n');
neuralNetwork = initNeuralNetwork(layerSizes);

% Load training data from file (sets X and y)
fprintf('\nLoading training data... \n');
load('ex4data1.mat');

% Train neural network
fprintf('\nTraining neural network... \n');
neuralNetwork = train(neuralNetwork, layerSizes, X, y, lambda, maxIterations);

% Test trained neural network
predictions = zeros(size(X,1), 1);
for i=1:size(X,1)
  predictions(i) = predict(neuralNetwork, layerSizes, X(i,:));
end
fprintf('\nTraining Set Accuracy: %f\n', mean(double(predictions == y)) * 100);
```
