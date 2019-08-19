% Predicts a result for input based on neuralNetwork
% layerSizes is a vector that determines the size of each layer (including the input and output layers) 
% without taking into consideration the bias for non-output layer
function p = predict(neuralNetwork, input, layerSizes)

  start = 1;
  for i=1:length(layerSizes)-1
    % Reshape Theta for the current layer
    finish = start - 1 + layerSizes(i+1) * (layerSizes(i) + 1);
    Theta = reshape(neuralNetwork(start:finish), ...
              layerSizes(i+1), (layerSizes(i) + 1));
    start = finish+1;

    % Add intercept term to input
    input = [1 input];

    % Compute input for the next layer
    input = sigmoid(Theta * input')';
  end

  [value, p] = max(input, [], 2);
end
