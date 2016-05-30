function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  h_theta = exp(theta' * X); % (num_classes-1) * m
  h_theta = [h_theta; ones(1, m)];  % num_classes * m
  p_theta = bsxfun(@rdivide, h_theta, sum(h_theta));  % num_classes * m
  h_theta = log(p_theta);
%   I = sub2ind(size(h_theta), y, 1:m);
%   f = -sum(sum(h_theta(I)));
  groundTruth = full(sparse(y, 1:m, 1)); % num_classes * m
  f = -sum(sum(groundTruth .* h_theta));
  g = -X * (groundTruth- p_theta)';  % n * num_classes
  g = g(:, 1:num_classes-1);
  g=g(:); % make gradient a vector for minFunc

