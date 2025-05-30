function [J grad] = costFunction(neuralNetwork, layerSizes, X, y, lambda)
  % Init some useful values
  m           = size(X, 1);
  numLayers   = length(layerSizes);
  costs       = 0;
  reg_costs   = 0;
  Identity    = eye(layerSizes(length(layerSizes)));
  Thetas      = cell(length(layerSizes)-1, 1);
  ThetasGrad  = cell(length(layerSizes)-1, 1);
  Activations = cell(length(layerSizes), 1);
  grad        = [];
  Activations{1} = X;

  start = 1;
  for i=1:length(layerSizes)-1
    % Reshape Theta for the current layer
    finish = start - 1 + layerSizes(i+1) * (layerSizes(i) + 1);
    Thetas{i} = reshape(neuralNetwork(start:finish), ...
              layerSizes(i+1), (layerSizes(i) + 1));
    ThetasGrad{i} = zeros(size(Thetas{i}));
    start = finish+1;

    % Setup Input by adding intercept terms to current Activations
    Input = [ones(size(Activations{i},1), 1) Activations{i}];

    % Set activations for the next layer
    Activations{i+1} = sigmoid(Input * Thetas{i}');

    % Compute regularization cost adjustment
    reg_costs += sum(sum(Thetas{i}(:, 2:end) .^ 2));
  end

  % Compute cost based on output layer
  for i=1:m
    row = -(Identity(y(i),:)) .* log(Activations{numLayers}(i,:)) - (1 - Identity(y(i),:)) .* log(1 - Activations{numLayers}(i,:));
    costs += sum(row);

    % Backpropagation
    delta = (Activations{numLayers}(i,:) - (Identity(y(i),:)))';

    for j=numLayers-1:-1:1
      Input = [1 Activations{j}(i, :)];
      ThetasGrad{j} += delta * Input;
      delta = (Thetas{j}' * delta) .* (Input .* (1-Input))';
      delta = delta(2:end);
    end
  end

  % Compute gradients for Theta
  for i=1:numLayers-1
    ThetaReg = Thetas{i};
    ThetaReg(:, 1) = 0;
    grad = [grad ; (ThetasGrad{i} + lambda * ThetaReg)(:)];
  end

  % Finalize computation of J and grad
  J = (costs/m) + (lambda/(2*m) * reg_costs);
  grad = grad / m;

end
