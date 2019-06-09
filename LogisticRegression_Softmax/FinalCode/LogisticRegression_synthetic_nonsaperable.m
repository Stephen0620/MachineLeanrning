clear,clc;
close all;

%% Synthetic Training Data
rng(108)
N=100; % Number of training samples
% Generating Class1
class1(:,1)=5+0.8*randn(N,1);

% Generating Class2
class2(:,1)=8.5+0.8*randn(N,1);

% Label
t =[zeros(N,1);ones(N,1)]; 

% X Designed Matrix
X = [class1;class2];

% Phi using RBF
phi_train=ones(size(X,1),1);
s=0.3;
for j=1:size(X,1)-1
    phi_train=[phi_train exp(-(X-X(j)).^2/(2*s^2))];
end

% Plot
figure;
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
legend('Class1','Class2');
hold off;
ylim([-0.2 1.2])
ylabel('Class')
xlabel('Samples')
title('The training synthetic dataset')
clear class1 class2;
%% Synthetic Testing Data
N=100; % Number of training samples
% Generating Class1
class1(:,1)=5+0.8*randn(N,1);

% Generating Class2
class2(:,1)=8.5+0.8*randn(N,1);

% Label
t_test =[zeros(N,1);ones(N,1)]; 

% X Designed Matrix
X_test = [class1;class2];

% Phi using RBF
phi_test=ones(size(X,1),1);
s=0.3;
for j=1:size(X,1)-1
    phi_test=[phi_test exp(-(X_test-X(j)).^2/(2*s^2))];
end
figure;
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
legend('Class1','Class2');
hold off;
ylim([-0.2 1.2])
ylabel('Class')
xlabel('Samples')
title('The testing synthetic dataset')

%% Gradient Descent
rho = 0.05;
maxIteration = 10000;
Lambda = 0.05;
disp('Training time of Logistic Regression on synthetic data with Gradient Descent:...')

tic
[WGradientDescend,Iteration] = GradientDescent(phi_train,t,Lambda,rho,maxIteration);
toc

% Get the Prediction
XW = phi_test*WGradientDescend;
Y = 1./(1+exp(-XW));
P = zeros(size(Y));
P(Y<0.5) = 0;
P(Y>0.5) = 1;

% Error
ErrorRate_Gradient = Error(P,t_test)
Mis = find((P~=t_test)==true);

% Plot
xAxis = [-10:0.1:9.9]';
phi_xAxis=ones(size(xAxis,1),1);
for j=1:size(X,1)-1
    phi_xAxis=[phi_xAxis exp(-(xAxis-X(j)).^2/(2*s^2))];
end
DecisionAxis=phi_xAxis*WGradientDescend;
yPlot = 1./(1+exp(-DecisionAxis));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
scatter(X_test(Mis,1),t_test(Mis),'rx','LineWidth',1.5);
plot(xAxis,yPlot);
ylim([0 1]);
xlim([min(class1) max(class2)]);
legend('Class1','Class2','MisClassified','y');
ylabel('Target / Prediction')
xlabel('Samples')
title(['Gradient Descent, MaxIterations = ', num2str(maxIteration)]);
hold off;

%% Stochastic
Epoch = 100;
Lambda = 0.05;

disp('Training time of Logistic Regression on synthetic data with Stochastic Gradient Descent:...')

tic
[WStochastic,y] = Stochastic(phi_train,t,Lambda,Epoch);

toc
% Get the Prediction
XW = phi_test*WStochastic;
Y = 1./(1+exp(-XW));
P = zeros(size(Y));
P(Y<0.5) = 0;
P(Y>0.5) = 1;

% Error
ErrorRate_Stochastic = Error(P,t_test)

Mis = find((P~=t_test)==true);

% Plot
xAxis = [-10:0.1:9.9]';
phi_xAxis=ones(size(xAxis,1),1);
for j=1:size(X,1)-1
    phi_xAxis=[phi_xAxis exp(-(xAxis-X(j)).^2/(2*s^2))];
end
DecisionAxis=phi_xAxis*WStochastic;
yPlot = 1./(1+exp(-DecisionAxis));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
scatter(X_test(Mis,1),t_test(Mis),'rx','LineWidth',1.5);
plot(xAxis,yPlot);
ylim([0 1]);
xlim([min(class1) max(class2)]);
legend('Class1','Class2','MisClassified','y');
ylabel('Target / Prediction')
xlabel('Samples')
title(['Stochastic GD, No. of Epochs = '  num2str(Epoch)]);
hold off;

%% Minibatch
Epoch = 100;
Batch = 10;
Beta = 0.1;
Lambda = 0.05;
disp('Training time of Logistic Regression on synthetic data with Minibatch:...')
tic
[WMini,y] = MiniBatch(phi_train,t,Lambda,Epoch,Batch,Beta);
toc

% Get the Prediction
XW = phi_test*WMini;
Y = 1./(1+exp(-XW));
P = zeros(size(Y));
P(Y<0.5) = 0;
P(Y>0.5) = 1;

% Error
ErrorRate_MiniBatch = Error(P,t_test)
Mis = find((P~=t_test)==true);

% Plot
xAxis = [-10:0.1:9.9]';
phi_xAxis=ones(size(xAxis,1),1);
for j=1:size(X,1)-1
    phi_xAxis=[phi_xAxis exp(-(xAxis-X(j)).^2/(2*s^2))];
end
DecisionAxis=phi_xAxis*WMini;
yPlot = 1./(1+exp(-DecisionAxis));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
scatter(X_test(Mis,1),t_test(Mis),'rx','LineWidth',1.5);
plot(xAxis,yPlot);
ylim([0 1]);
xlim([min(class1) max(class2)]);
legend('Class1','Class2','MisClassified','y');
ylabel('Target / Prediction')
xlabel('Samples')
title(['GD with Mini batch, No. of Epochs = '  num2str(Epoch),', Minibatch size= ',num2str(Batch)]);
hold off;