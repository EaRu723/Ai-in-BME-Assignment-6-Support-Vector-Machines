%% PA6 Bonus

%% Initialization
clear ; close all; clc

%% =============== Using SVM on PA2 Data ================
%  Loading in the Pa 2 data
%

fprintf('Loading Data ...\n')

% 60% of the data used for training, and 20% for CV, other 20% for testing
data = load('PA2data1.txt');

[m,n] = size(data);
per = 0.6;    
per2 = 0.2;
ind = randperm(m);
data_training = data(ind(1:round(per*m)),:);
data_CV = data(ind(round(per*m)+1:round((1-per2)*m)),:);
data_test = data(ind(round((1-per2)*m)+1:end),:);

% naming the y parameter
y = data_training(:,end);

X = data_training(:,1:end-1);
% Initialize CV sets
y_val = data_CV(:,end);
X_val = data_CV(:,1:end-1);
% Initialize test sets
y_test = data_test(:,end);
X_test = data_test(:,1:end-1);

% Normalize data using the ame mu and sigma
mean = mean(X);
sigma = std(X);
X = (X - mean) ./ sigma;
X_val = (X_val - mean) ./ sigma;
X_test = (X_test - mean) ./sigma;

%% ========== Training SVM with RBF Kernel ==========

% Try different SVM Parameters here
[C, sigma, minError] = dataset3Params(X, y, X_val, y_val);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
fprintf(['The minimal error = %f: with C = %f and sigma = %f\n'], minError, C, sigma);

% Computing accuracy of prediction
pred = svmPredict(model, X_test);
prediction_accuracy = 100 * mean(double(pred == y_test));
fprintf(['The accuracy of the prediction = %f\n'], prediction_accuracy);