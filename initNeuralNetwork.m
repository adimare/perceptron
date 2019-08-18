% Initializes and returns an unrolled neural network
% layerSizes is a vector that determines the size of each layer (not including the bias for non-output layers) 
function unrolledNeuralNetwork = initNeuralNetwork(layerSizes)
  unrolledNeuralNetwork = []

  epsilon_init = 0.12;
  for i=1:length(layerSizes)-1
    Theta = rand(layerSizes(i+1), layerSizes(i)+1) * 2 * epsilon_init - epsilon_init;
    unrolledNeuralNetwork = [unrolledNeuralNetwork; Theta(:)];
  end
end
