function J = costFunction(neuralNetwork, layerSizes, X, y, lambda)
  % Init Input
  Input = X;

  % Init 
  m = size(X, 1);

  start = 1;
  for i=1:length(layerSizes)-1
    % Reshape Theta for the current layer
    finish = start - 1 + layerSizes(i+1) * (layerSizes(i) + 1);
    Theta = reshape(neuralNetwork(start:finish), ...
              layerSizes(i+1), (layerSizes(i) + 1));
    start = finish+1;

    % Add intercept terms to Input
    Input = [ones(size(Input,1), 1) Input];

    % Compute input for the next layer
    Input = sigmoid(Input * Theta');
  end

  Identity = eye(layerSizes(length(layerSizes)));
  J = 0
  for i=1:m
    row = -(Identity(y(i),:)) .* log(Input(i,:)) - (1 - Identity(y(i),:)) .* log(1 - Input(i,:));
    J += sum(row);
  end

  J = J/m;
end
