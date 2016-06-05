function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
switch ei.activation_fun
    case 'logistic'
        act_fun = @sigmoid;
        derivate_fun = @derivate_sigmoid;
    case 'tanh'
        act_fun = @tanh_act;
        derivate_fun = @derivate_tanh;
    case 'relu'
        act_fun = @relu;
        derivate_fun = @derivate_relu;
    otherwise
        warning('activation type not defined.')
end

hAct{1} = data; % input  numFeatures * numCases
for i = 1 : numHidden
    % stack{i}.W   size = hidden{i} * pre_hidden 
    z = bsxfun(@plus, stack{i}.W * hAct{i}, stack{i}.b);
    % hAct{i+1} = activate(z, activation_fun);
    hAct{i+1} = act_fun(z);
end
% hAct{numHidden+1} is the output of the final hidden layer
% output layer(softmax)
h = bsxfun(@plus, stack{numHidden+1}.W * hAct{numHidden+1}, stack{numHidden+1}.b);
h = bsxfun(@minus, h, max(h, [], 1));
h = exp(h);  % numClasses * numCases
pred_prob = bsxfun(@rdivide, h, sum(h));  % numClasses * numCases

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
[~, numCases] = size(data);
groundTruth = full(sparse(labels, 1:numCases, 1));  % numClasses * numCases
% notice the cost J in tutorial is not so good, use mean may be better
% and there is a write error in the denote
ceCost = -mean(sum(groundTruth .* log(pred_prob))); 
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
gradStack{end}.W = -1/numCases * (groundTruth - pred_prob) * hAct{end}'; % softmax, 
% gradStack{end}.b = -1/numCases * sum(groundTruth - pred_prob, 2);
gradStack{end}.b = -mean(groundTruth - pred_prob, 2);

delta = cell(numHidden + 1, 1);
delta{numHidden+1} = -stack{numHidden+1}.W' * (groundTruth - pred_prob) .* derivate_fun(hAct{numHidden+1});  % numClasses * numCases

for layer = numHidden : -1 : 1
    delta{layer} = (stack{layer}.W' * delta{layer+1}) .* derivate_fun(hAct{layer});
end
for layer = numHidden : -1 : 1
    gradStack{layer}.W = delta{layer+1} * hAct{layer}' / numCases;
    gradStack{layer}.b = mean(delta{layer+1}, 2);
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wCost = 0;
for i = 1 : numel(stack)
    wCost = sum(sum(stack{i}.W .^ 2));
    gradStack{i}.W = gradStack{i}.W + ei.lambda * stack{i}.W;
end

cost = ceCost + ei.lambda * wCost;


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end


function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function y = tanh_act(x)
    y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
end

function y = relu(x)
    y = max(x, 0);
end

function y = derivate_sigmoid(a) % a = sigmoid(z), is the activation
    y = a .* (1-a);
end

function y = derivate_tanh(a) % a = tanh_act(z), is the activation
    y = 1 - a .^ 2 ;
end

function y = derivate_relu(z)
    y = zeros(size(z));
    y(z> 0) = 1;
end
% function y = activate(x, acttype)
%     switch acttype
%         case 'logistic'
%             y = 1 ./ (1 + exp(-x));
%         case 'tanh'
%             y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));
%         case 'relu'
%             y = max(x, 0);
%         otherwise
%             warning('activation type not defined.')
%     end
% end

