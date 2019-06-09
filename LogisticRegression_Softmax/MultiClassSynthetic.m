clear,clc;
close all;

%% Synthetic Data
N=100; % Number of training samples
% Generating Class1
class1(:,1)=0.3*randn(N,1);
% class1(:,2)=6+0.8*randn(N,1);

% Generating Class2
class2(:,1)=2+0.3*randn(N,1);
% class2(:,2)=9.5+0.8*randn(N,1);

% Generating Class3
class3(:,1)=4+0.3*randn(N,1);
% class2(:,2)=9.5+0.8*randn(N,1);

% X Designed Matrix
X = [class1;class2;class3];
X = [X,ones(size(X,1),1)];

K = 3;
T = zeros(size(X,1),K);
T(1:100,1) = 1;
T(101:200,2) = 1;
T(201:300,3) = 1;

Label(1:100) = 1;
Label(101:200) = 2;
Label(201:300) = 3;

figure;
hold on;
scatter(class1(:,1),Label(1:100));
scatter(class2(:,1),Label(101:200));
scatter(class3(:,1),Label(201:300));
legend('Class1','Class2','class3');
hold off;

%% Logistic Regression using Gradient Descend
w = zeros(size(X,2),K)';    % Initial Guess
rho = 0.001;
maxIteration = 10000;
Y = zeros(size(X,1),K);

for iteration=1:maxIteration
    wPrev = w;
    for i=1:size(X,1)
        XW = exp(w*X(i,:)');
        Denominator = sum(exp(w*X(i,:)'));
        Y(i,:) = XW./Denominator;        
    end
    for i=1:K
        w(i,:) = w(i,:) - rho.*sum((Y(:,i)-T(:,i)).*X);
    end
    if sum(norm(wPrev-w))<1e-6
        break;
    end
end    

for i=1:size(X,1)
    XW = exp(w*X(i,:)');
    Denominator = sum(exp(w*X(i,:)'));
    Y(i,:) = XW./Denominator;        
end

%% Predict
[~,P] = max(Y,[],2);