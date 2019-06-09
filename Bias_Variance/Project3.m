clear,clc;
close all;

rng(104);
%% Generate the training set
L=100;  % Number of data
N=25;   % Number of samples in each data
D=cell(L,1);    % Cell for storing each data
t=cell(L,1);    % Cell for storing each target
x_original=rand(N,1);

% Generating Data
for i=1:L
    N=25;
%     x_original=rand(N,1);
    Epsilon=randn(N,1)*0.3;
    t{i}=sin(2*pi*x_original)+Epsilon;
    D{i}=x_original;
end

%% Generate Phi Matrix
s=0.1;
Phi = cell(L,1);
for i=1:L
   Phi{i} = Gaussian_Phi(D{i},D{i},s); 
end

%% Closed Form
W=cell(L,1);    % Cell for storing the weight vectors of each data set
Lambda_ln=(-3.5:0.1:3); % Different Lambdas
Lambda=exp(Lambda_ln);
Weight=zeros(N,numel(Lambda));

% Get the weight vecotr with different lambdas for different data set
for i=1:L
    for j=1:numel(Lambda)
        % Regularization
        Weight(:,j)=inv(Phi{i}'*Phi{i}+Lambda(j).*eye(size(Phi{i})))*Phi{i}'*t{i}; 
    end
    W{i}=Weight;
end

%% Generate testing set
N_test=1000; % Number of samples of testing set
x_test_original=rand(N_test,1);
Epsolon_test=randn(N_test,1)*0.3;
t_test=sin(2*pi*x_test_original)+Epsolon_test; 

%% Getting the Phi for testing set
% Get the Phi for different dataset
Phi_test=cell(L,1);
for i=1:L
    Phi_test{i}=Gaussian_Phi(x_test_original,D{i},s);
end

%% Prediction
t_predict=zeros(N_test,numel(Lambda),L);    % Predicted target
predict=zeros(N_test,numel(Lambda));  

% First dimension: Number of samples
% Second dimension: Number of Lambdas
% Third dimension: Number of data sets
for i=1:L
    for j=1:numel(Lambda)
        Weight=W{i}(:,j);
        predict(:,j)=Phi_test{i}*Weight;
    end
    t_predict(:,:,i)=predict;
end

% Calculate bias square
f_bar=mean(t_predict,3);    % f_bar = Mean over the third dimension
true_prediction=sin(2*pi*x_test_original);
bias_square=mean((f_bar-true_prediction).^2,1); 

% Calculate Variance
Var=mean((t_predict-f_bar).^2,3);   % Variance = mean over third dimension
Var=mean(Var,1);

%% Test error
% Sum over first dimension then mean over third dimension
Errors = mean(sqrt(sum((t_predict-t_test).^2,1)./N_test),3);    

%% Plotting
figure; 
hold on;
plot(Lambda_ln,bias_square);
plot(Lambda_ln,Var);
plot(Lambda_ln,bias_square+Var);
plot(Lambda_ln,Errors);
xlabel('ln\lambda');
legend('(bias)^2','variance','(bias)^2+variance','Test Error');
legend('Location','best');
hold off;

%% Plot Model
x=(0:0.01:1);
x_Phi=Gaussian_Phi(x,D{1},s);
best_lambda=find(Errors==min(Errors));

figure; 
hold on;
plot(x,x_Phi*W{1}(:,1)); 
plot(x,x_Phi*W{1}(:,end)); 
plot(x,x_Phi*W{1}(:,best_lambda)); % Lambda that produces minimun test error
plot(x,sin(2*pi*x)); % true prediction
L1=['ln\lambda=' num2str(Lambda_ln(1))];
L2=['ln\lambda=' num2str(Lambda_ln(end))];
L3=['best Model \lambda=' num2str(Lambda_ln(best_lambda))];
legend(L1,L2,L3,'true prediction');
hold off;

% LB HV Overfitting
figure;
title('Low Bias, High Var');
hold on;
for i=1:L
    plot(x,x_Phi*W{i}(:,1));
end
hold off;

% Best model
figure;
title('Best Model');
hold on;
for i=1:L
    plot(x,x_Phi*W{i}(:,best_lambda));
end
hold off;

% HB LV Undrfitting
figure;
title('High Bias, Low Var');
hold on;
for i=1:L
    plot(x,x_Phi*W{i}(:,end));
end
hold off;
