% Creates a small neural network to check the backpropagation gradients
function checkNNGradients(lambda)
  if ~exist('lambda', 'var') || isempty(lambda)
      lambda = 0;
  end

  inputLayerSize  = 3;
  hiddenLayerSize = 5;
  outputLayerSize = 3;
  layerSizes = [inputLayerSize hiddenLayerSize outputLayerSize];
  m = 5;

  % Generate some 'random' test data
  Theta1 = debugInitializeWeights(hiddenLayerSize, inputLayerSize);
  Theta2 = debugInitializeWeights(outputLayerSize, hiddenLayerSize);
  % Reusing debugInitializeWeights to generate X
  X  = debugInitializeWeights(m, inputLayerSize - 1);
  y  = 1 + mod(1:m, outputLayerSize)';

  % Unroll parameters
  smallNeuralNetwork = [Theta1(:) ; Theta2(:)];

  % Short hand for cost function
  costFunc = @(p) costFunction(p, layerSizes, X, y, lambda);

  [cost, grad] = costFunc(smallNeuralNetwork);
  numgrad = computeNumericalGradient(costFunc, smallNeuralNetwork);

  % Visually examine the two gradient computations. The two columns should be very similar. 
  disp([numgrad grad]);
  fprintf(['The above two columns should be very similar.\n' ...
           '(Left-Numerical Gradient, Right-Analytical Gradient)\n\n']);

  % Evaluate the norm of the difference between two solutions.  
  % The diff below should be less than 1e-9
  diff = norm(numgrad-grad)/norm(numgrad+grad);

  fprintf(['The relative difference should be small (less than 1e-9). \n' ...
           '\nRelative Difference: %g\n'], diff);

end
