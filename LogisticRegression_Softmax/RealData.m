clear,clc;
close all;

%% Read the Leaf data
S = textread('Type.txt','%s');
D = xlsread('Data.xls');
Species = unique(S);
K = numel(Species);

for i=1:K
    idx = find(strcmp(S,Species{i})==1);
    T(idx,i) = 1;
    Label(idx) = i;
end

% Normalize
D = (D - mean(D))./std(D);

X = [D(:,4:6),ones(size(D,1),1)]; % Designed matrix

%% Logistic Regression using Gradient Descend
w = zeros(size(X,2),K)';    % Initial Guess
rho = 0.005;
maxIteration = 1000000;
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
