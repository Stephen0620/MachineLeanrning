clear,clc;
close all;

%% Synthetic Data
N=100; % Number of training samples
% Generating Class1
class1(:,1)=1+0.8*randn(N,1);
% class1(:,2)=6+0.8*randn(N,1);

% Generating Class2
class2(:,1)=8.5+0.8*randn(N,1);
% class2(:,2)=9.5+0.8*randn(N,1);

% t 
t =[zeros(N,1);ones(N,1)]; 

% X Designed Matrix
X = [class1;class2];
X = [X,ones(size(X,1),1)];

% Plot
figure;
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
legend('Class1','Class2');
hold off;

figure; plot((1:numel(t)),t);

%% Logistic Regression using Stochastic Gradient Descend
rho = 1/(N);
w = zeros(size(X,2),1); %Initial guess 
idx = randperm(size(X,1));
XRandom = X(idx,:);
tRandom = t(idx,:);
Epoch = 100;

for j=1:Epoch
    for i=1:2*N
        y = 1./(1+exp(-XRandom(i,:)*w));
        w = w-((rho).*(y-tRandom(i)).*XRandom(i,:))';
    end
end

XW = X*w;
y = 1./(1+exp(-XW));

%% plot of the decision boundary of stochastic gradient descend
xAxis = [1:0.01:12]';
decision = [xAxis,ones(numel(xAxis),1)]*w;
yPlot = 1./(1+exp(-decision));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
% plot(xAxis,decision);
plot(xAxis,yPlot);
ylim([0 1]);
legend('Class1','Class2','Decision Boundary','y');
hold off;
figure;
plot(y);
