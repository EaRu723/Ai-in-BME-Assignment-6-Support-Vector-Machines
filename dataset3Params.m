function [C, sigma, minError] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return C, sigma, and minError correctly.


minError = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C = [.01, .03, .1, .3, 1, 3, 10, 30];
sigma = [.01, .03, .1, .3, 1, 3, 10, 30];
Error = zeros(length(C),length(sigma));
for i = 1:length(C)
    for j = 1:length(sigma)
model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
predictions = svmPredict(model, Xval);
Error(i,j) = mean(double(predictions ~= yval));
    end
end

[min_val,idx]=min(Error(:));
[row,col]=ind2sub(size(Error),idx);

minError = min_val;
C = C(row);
sigma = sigma(col);


% =========================================================================

end
