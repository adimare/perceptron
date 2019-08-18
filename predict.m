% Predicts a result for input based on unrolledNeuralNetwork
% layerSizes is a vector that determines the size of each layer (including the input and output layers) without taking into consideration the bias for non-output layers
function p = predict(input, unrolledNeuralNetwork, layerSizes)

  start = 1;
  for i=1:length(layerSizes)-1
    % Unroll Theta for the current layer
    finish = start - 1 + layerSizes(i+1) * (layerSizes(i) + 1);
    Theta = reshape(unrolledNeuralNetwork(start:finish), ...
              layerSizes(i+1), (layerSizes(i) + 1));
    start = finish+1;

    % Add intercept term to input
    input = [1 input];

    % Compute input for the next layer
    input = sigmoid(Theta * input')';
  end

  [value, p] = max(input, [], 2);
end
