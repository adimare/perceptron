% Initializes and returns an unrolled neural network
% layerSizes is a vector that determines the size of each layer (not including the bias for non-output layers) 
function unrolledNeuralNetwork = initNeuralNetwork(layerSizes)
  unrolledNeuralNetwork = [];

  % Uses a small epsilon to generate initial values
  epsilon_init = 0.12;
  for i=1:length(layerSizes)-1
    unrolledNeuralNetwork = [unrolledNeuralNetwork; Theta(:)];
  end
end
