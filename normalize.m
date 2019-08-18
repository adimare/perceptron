% Returns a normalized version of X, a vector containing the means of each feature in X,
% and a vector containing the standard deviation of each feature in X
function [X_norm, mu, sigma] = normalize(X)
  [m, n] = size(X);
  mu = zeros(1, n);
  sigma = zeros(1, n);

  for i=1:n
    mu(i) = mean( X(:,i) );
    sigma(i) = std( X(:,i) );

    X_norm(:,i) = (X(:,i) - mu(i)) / sigma(i);
  end
end