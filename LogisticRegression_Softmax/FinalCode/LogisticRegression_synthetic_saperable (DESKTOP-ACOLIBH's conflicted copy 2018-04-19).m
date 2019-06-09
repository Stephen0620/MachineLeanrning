clear,clc;
close all;

rng(108)
%% Synthetic Training Data
N=100; % Number of training samples
% Generating Class1
class1(:,1)=3+0.8*randn(N,1);

% Generating Class2
class2(:,1)=8.5+0.8*randn(N,1);

% Label
t =[zeros(N,1);ones(N,1)]; 

% X Designed Matrix
X = [class1;class2];
X_train = [X,ones(size(X,1),1)];

% Plot
figure;
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
legend('Class1','Class2');
xlabel('Samples');
ylabel('Target / Prediction');
hold off;
clear class1 class2;
%% Synthetic Testing Data
N=100; % Number of training samples
% Generating Class1
class1(:,1)=3+0.8*randn(N,1);

% Generating Class2
class2(:,1)=8.5+0.8*randn(N,1);

% Label
t_test =[zeros(N,1);ones(N,1)]; 

% X Designed Matrix
X_test = [class1;class2];
X_test = [X_test,ones(size(X_test,1),1)];

%% Gradient Descent
rho = 0.5;
maxIteration = 10000;
Lambda = 0;
[WGradientDescend,Iteration] = GradientDescent(X_train,t,Lambda,rho,maxIteration);
WGradientDescend

% Get the Prediction
XW = X_test*WGradientDescend;
Y = 1./(1+exp(-XW));
P = zeros(size(Y));
P(Y<0.5) = 0;
P(Y>0.5) = 1;

% Error
ErrorRate_Gradient = Error(P,t_test)
Mis = find((P~=t_test)==true);

% Plot
xAxis = [-10:0.1:9.9]';
AXIS = [xAxis,ones(size(xAxis))];
decision = -WGradientDescend(2)./WGradientDescend(1);
yPlot = 1./(1+exp(-AXIS*WGradientDescend));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
scatter(X_test(Mis,1),t_test(Mis),'rx','LineWidth',1.5);
idx = (0:0.01:1);
plot(ones(size(idx))*decision,idx);
plot(xAxis,yPlot);
ylim([0 1]);
xlim([min(class1) max(class2)]);
legend('Class1','Class2','MisClassified','Decision Boundary','y');
xlabel('Samples');
ylabel('Target / Prediction');
title(['Gradient Descent with iteration = ', num2str(Iteration)]);
hold off;

%% Stochastic
Epoch = 400;
Lambda = 0.05;
[WStochastic,y] = Stochastic(X_train,t,Lambda,Epoch);

% Get the Prediction
XW = X_test*WStochastic;
Y = 1./(1+exp(-XW));
P = zeros(size(Y));
P(Y<0.5) = 0;
P(Y>0.5) = 1;

% Error
ErrorRate_Stochastic = Error(P,t_test)
Mis = find((P~=t_test)==true);

% Plot
xAxis = [-10:0.1:9.9]';
AXIS = [xAxis,ones(size(xAxis))];
decision = -WStochastic(2)./WStochastic(1);
yPlot = 1./(1+exp(-AXIS*WStochastic));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
scatter(X_test(Mis,1),t_test(Mis),'rx','LineWidth',1.5);
idx = (0:0.01:1);
plot(ones(size(idx))*decision,idx);
plot(xAxis,yPlot);
ylim([0 1]);
xlim([min(class1) max(class2)]);
legend('Class1','Class2','MisClassified','Decision Boundary','y');
xlabel('Samples');
ylabel('Target / Prediction');
title(['Stochastic, No. of Epochs = '  num2str(Epoch)]);
hold off;

%% Minibatch
Epoch = 1000;
Batch = 5;
Beta = 0.1;
Lambda = 0;
[WMini,y] = MiniBatch(X_train,t,Lambda,Epoch,Batch,Beta);

% Get the Prediction
XW = X_test*WMini;
Y = 1./(1+exp(-XW));
P = zeros(size(Y));
P(Y<0.5) = 0;
P(Y>0.5) = 1;

% Error
ErrorRate_MiniBatch = Error(P,t_test)
Mis = find((P~=t_test)==true);

% Plot
xAxis = [-10:0.1:9.9]';
AXIS = [xAxis,ones(size(xAxis))];
decision = -WMini(2)./WMini(1);
yPlot = 1./(1+exp(-AXIS*WMini));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
scatter(X_test(Mis,1),t_test(Mis),'rx','LineWidth',1.5);
idx = (0:0.01:1);
plot(ones(size(idx))*decision,idx);
plot(xAxis,yPlot);
ylim([0 1]);
xlim([min(class1) max(class2)]);
legend('Class1','Class2','MisClassified','Decision Boundary','y');
xlabel('Samples');
ylabel('Target / Prediction');
title(['Mini batch, No. of Epochs = '  num2str(Epoch),', Minibatch size = ',num2str(Batch)]);
hold off;