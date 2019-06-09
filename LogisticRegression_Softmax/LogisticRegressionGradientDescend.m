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

%% Logistic Regression using Gradient Descend
WGradientDescend = zeros(size(X,2),1);    % Initial Guess
rho = 0.05;
maxIteration = 10000000;
% gradient = zeros(maxIteration,size(WGradientDescend,1));

WPrev = WGradientDescend;
for i=1:maxIteration
%     WGradientDescend = WGradientDescend - rho.*gradient(i,:)';
    if i>=2&&norm(WPrev-WGradientDescend)<1e-6
        break;
    else
        WPrev = WGradientDescend;
        XW = X*WGradientDescend;
        y = 1./(1+exp(-XW));
        gradient = sum((y-t).*X);
        WGradientDescend = WGradientDescend - rho.*gradient';
    end
end

%% plot of the decision boundary of gradient descend
xAxis = [1:0.01:12]';
decision = [xAxis,ones(numel(xAxis),1)]*WGradientDescend;
yPlot = 1./(1+exp(-decision));
figure; 
hold on;
scatter(class1(:,1),t(1:N));
scatter(class2(:,1),t(N+1:2*N));
plot(xAxis,decision);
plot(xAxis,yPlot);
ylim([0 1]);
legend('Class1','Class2','Decision Boundary','y');
hold off;
